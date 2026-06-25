#!/usr/bin/env python

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
LeRobot Dataset Loader

A simplified, clean implementation for loading LeRobot datasets with video support.
This module provides the core functionality for loading episodes from LeRobot format datasets,
handling metadata parsing, video decoding, and data preprocessing for VLA training.

The LeRobotEpisodeLoader serves as the foundation for higher-level dataset classes,
providing episode-level data access with support for multi-modal data including:
- Video frames from multiple camera views
- Proprioceptive state information
- Action sequences
- Language instructions/annotations

Returns messages with VLAStepData as defined in types.py.
"""

from collections import OrderedDict, defaultdict
import json
import logging
from pathlib import Path
import random
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

from gr00t.data.types import ModalityConfig
from gr00t.utils.initial_actions import INITIAL_ACTIONS_FILENAME, load_initial_actions
from gr00t.utils.video_utils import VideoReaderPool, get_frames_by_indices


# LeRobot standard metadata filenames
LEROBOT_META_DIR_NAME = "meta"
LEROBOT_INFO_FILENAME = "info.json"
LEROBOT_EPISODES_FILENAME = "episodes.jsonl"
LEROBOT_TASKS_FILENAME = "tasks.jsonl"
LEROBOT_MODALITY_FILENAME = "modality.json"
LEROBOT_STATS_FILE_NAME = "stats.json"
LEROBOT_RELATIVE_STATS_FILE_NAME = "relative_stats.json"

# LeRobot v3.0 consolidates per-episode files into multi-episode chunk files and
# stores episode/task metadata as parquet (vs. JSONL in v2.x). These live under
# meta/ and replace episodes.jsonl / tasks.jsonl when codebase_version >= v3.0.
LEROBOT_V30_EPISODES_DIR_NAME = "episodes"
LEROBOT_V30_TASKS_FILENAME = "tasks.parquet"

ALLOWED_MODALITIES = ["video", "state", "action", "language", "mask"]
DEFAULT_COLUMN_NAMES = {
    "state": "observation.state",
    "action": "action",
}

LANG_KEYS = ["task", "sub_task"]


def _rec_defaultdict() -> defaultdict:
    """Factory that creates an infinitely nestable defaultdict."""
    return defaultdict(_rec_defaultdict)


def _to_plain_dict(tree):
    """Recursively turn a (nested) defaultdict into a regular dict."""
    if isinstance(tree, defaultdict):
        return {k: _to_plain_dict(v) for k, v in tree.items()}
    return tree


class LeRobotEpisodeLoader:
    """
    Episode-level data loader for LeRobot format datasets.

    This class handles the loading and preprocessing of individual episodes from LeRobot datasets.
    It manages metadata parsing, video decoding, and data extraction across multiple modalities
    (video, state, action, language) while maintaining compatibility with the VLA training pipeline.

    Key responsibilities:
    - Parse LeRobot metadata files (info.json, episodes.jsonl, etc.)
    - Load and decode video data using configurable backends
    - Extract and process multi-modal data according to modality configurations
    - Provide dataset statistics for normalization
    - Handle initial action loading for policy initialization

    Args:
        dataset_path: Path to dataset root directory containing meta/ and data files
        modality_configs: Dictionary mapping modality names to ModalityConfig objects
                         that specify temporal sampling and data keys to load
        video_backend: Video decoding backend ('torchcodec', 'decord', etc.)
        video_backend_kwargs: Additional arguments for the video backend

    Example:
        >>> loader = LeRobotEpisodeLoader(
        ...     dataset_path="/path/to/lerobot_dataset",
        ...     modality_configs={
        ...         "video": ModalityConfig(delta_indices=[0], modality_keys=["front_cam"]),
        ...         "state": ModalityConfig(delta_indices=[0], modality_keys=["joint_positions"]),
        ...         "action": ModalityConfig(
        ...             delta_indices=list(range(16)), modality_keys=["joint_velocities"]
        ...         ),
        ...     },
        ... )
        >>> episode_data = loader[0]  # Load first episode as DataFrame
    """

    def __init__(
        self,
        dataset_path: str | Path,
        modality_configs: dict[str, ModalityConfig],
        video_backend: str = "torchcodec",
        video_backend_kwargs: dict[str, Any] | None = None,
        data_cache_size: int | None = None,
        video_cache_size: int | None = None,
    ) -> None:
        """
        Initialize LeRobot episode loader with dataset path and modality configurations.

        The initialization process involves:
        1. Loading all metadata files from the dataset
        2. Parsing and validating modality configurations
        3. Computing effective episode lengths based on action horizon
        4. (v3.0 only) Setting up the per-file parquet/video caches

        Args:
            data_cache_size: Max v3.0 data parquet tables to keep cached.
                ``None`` defaults to the data-file count (capped). Ignored for v2.x.
            video_cache_size: Max v3.0 video decoders to keep cached.
                ``None`` defaults to the video-file count (capped). Ignored for v2.x.
        """
        self.dataset_path = Path(dataset_path)
        self.video_backend = video_backend
        self.video_backend_kwargs = video_backend_kwargs

        if not self.dataset_path.is_dir():
            raise FileNotFoundError(f"Dataset path does not exist: {self.dataset_path}")

        # Load metadata files and parse dataset structure
        self._load_metadata()

        # Set up modality configs after metadata is loaded
        self.modality_configs = self._parse_and_validate_modality_configs(modality_configs)

        # Compute effective episode lengths accounting for action horizon
        self.episode_lengths = self.get_episode_lengths()

        # Per-file I/O caches keyed by file, so v3.0's many-episodes-per-file
        # layout reads each file once instead of once per episode. Inert on v2.x.
        self._table_cache: "OrderedDict[tuple[int, int], Any]" = OrderedDict()
        self._video_pool: VideoReaderPool | None = None
        if self.is_v30:
            self._init_v30_caches(data_cache_size, video_cache_size)

    def _load_metadata(self) -> None:
        """
        Load all metadata files including dataset statistics.

        Parses the standard LeRobot metadata structure:
        - info.json: Dataset configuration and file patterns
        - episodes.jsonl: Per-episode metadata (length, timestamps, etc.)
        - tasks.jsonl: Task descriptions and mappings
        - modality.json: Modality structure and data layout
        - stats.json: Dataset statistics for normalization
        """
        meta_dir = self.dataset_path / LEROBOT_META_DIR_NAME

        # Load dataset configuration
        info_path = meta_dir / LEROBOT_INFO_FILENAME
        with open(info_path, "r") as f:
            self.info_meta = json.load(f)

        # Detect the LeRobot dataset codebase version
        self.codebase_version = str(self.info_meta.get("codebase_version", "v2.1"))
        self.is_v30 = self._parse_major_version(self.codebase_version) >= 3

        if self.is_v30:
            self.episodes_metadata = self._load_episodes_metadata_v30(meta_dir)
            self.tasks_map = self._load_tasks_v30(meta_dir)
        else:
            # Load episode metadata (one episode per line)
            episodes_path = meta_dir / LEROBOT_EPISODES_FILENAME
            with open(episodes_path, "r") as f:
                self.episodes_metadata = [json.loads(line) for line in f]

            # Load task descriptions and create mapping
            tasks_path = meta_dir / LEROBOT_TASKS_FILENAME
            with open(tasks_path, "r") as f:
                tasks_data = [json.loads(line) for line in f]
                self.tasks_map = {task["task_index"]: task["task"] for task in tasks_data}

        # Index episode records by their episode_index
        self._episode_by_index = {int(ep["episode_index"]): ep for ep in self.episodes_metadata}

        # Load modality structure information
        modality_path = meta_dir / LEROBOT_MODALITY_FILENAME
        with open(modality_path, "r") as f:
            self.modality_meta = json.load(f)

        # Load dataset statistics for normalization
        stats_path = meta_dir / LEROBOT_STATS_FILE_NAME
        assert stats_path.exists(), (
            f"{stats_path} does not exist for {self.dataset_path}, please use gr00t/data/stats.py to generate it"
        )
        with open(stats_path, "r") as f:
            self.stats = json.load(f)

        relative_stats_path = meta_dir / LEROBOT_RELATIVE_STATS_FILE_NAME
        if relative_stats_path.exists():
            with open(relative_stats_path, "r") as f:
                relative_stats = json.load(f)
            # Drop the cache-invalidation sidecar written by gr00t.data.stats
            # (mirrors STATS_FINGERPRINTS_KEY there). Consumers index by
            # feature name and would otherwise treat it as a stats group.
            relative_stats.pop("__fingerprints__", None)
            self.stats["relative_action"] = relative_stats

        # Extract key configuration parameters
        self.feature_config = self.info_meta.get("features", {})
        self.data_path_pattern = self.info_meta["data_path"]
        self.video_path_pattern = self.info_meta.get("video_path")
        self.mask_path_pattern = self.info_meta.get("mask_path")
        self.chunk_size = self.info_meta["chunks_size"]
        self.fps = self.info_meta.get("fps", 30)

    @staticmethod
    def _parse_major_version(codebase_version: str) -> int:
        """Extract the integer major version from a ``vX.Y`` codebase string.
        """
        digits = codebase_version.lstrip("vV").split(".")[0]
        try:
            return int(digits)
        except ValueError:
            return 2

    def _load_episodes_metadata_v30(self, meta_dir: Path) -> list[dict[str, Any]]:
        """Load consolidated per-episode metadata rows from ``meta/episodes/``.

        v3.0 stores episode metadata as parquet (one row per episode) holding the
        data-file location (``data/chunk_index``, ``data/file_index``,
        ``dataset_from_index``, ``dataset_to_index``) and, per video key, the
        containing file and its time span. The heavy per-episode ``stats/*``
        columns are skipped — they are not needed for loading and inflate memory.
        """
        episodes_dir = meta_dir / LEROBOT_V30_EPISODES_DIR_NAME
        pq_paths = sorted(episodes_dir.glob("chunk-*/file-*.parquet"))
        if not pq_paths:
            raise FileNotFoundError(
                f"No episode parquet files found under {episodes_dir} for v3.0 dataset "
                f"{self.dataset_path}"
            )
        records: list[dict[str, Any]] = []
        for pq_path in pq_paths:
            schema_names = pq.ParquetFile(pq_path).schema_arrow.names
            columns = [name for name in schema_names if not name.startswith("stats/")]
            records.extend(pq.read_table(pq_path, columns=columns).to_pylist())
        records.sort(key=lambda record: int(record["episode_index"]))
        return records

    def _load_tasks_v30(self, meta_dir: Path) -> dict[int, str]:
        """Load the task-index -> task-string map from ``meta/tasks.parquet``.

        v3.0 stores tasks as parquet with an integer ``task_index`` column. The
        task string may be either a regular ``task`` column or the (named) index,
        depending on the writer; both layouts are normalized here.
        """
        tasks_df = pq.read_table(meta_dir / LEROBOT_V30_TASKS_FILENAME).to_pandas()
        if tasks_df.index.name == "task":
            tasks_df = tasks_df.reset_index()
        return {int(row["task_index"]): str(row["task"]) for _, row in tasks_df.iterrows()}

    def _init_v30_caches(self, data_cache_size: int | None, video_cache_size: int | None) -> None:
        """Set up the v3.0 per-file parquet table cache and video decoder pool.

        Precomputes the projected data columns and each file's base global row
        index (to map an episode's global range to a within-file slice). Cache
        sizes default to the file counts (capped) for order-independent reuse.
        """
        self._data_columns = self._compute_needed_data_columns()

        # Base (minimum global row index) per data file → within-file offsets.
        self._file_row_base: dict[tuple[int, int], int] = {}
        for ep in self.episodes_metadata:
            key = (int(ep["data/chunk_index"]), int(ep["data/file_index"]))
            frm = int(ep["dataset_from_index"])
            cur = self._file_row_base.get(key)
            self._file_row_base[key] = frm if cur is None else min(cur, frm)

        n_data_files = len(self._file_row_base)
        self._table_cache_size = (
            data_cache_size if data_cache_size is not None else max(1, min(n_data_files, 64))
        )

        n_video_files = self._count_v30_video_files()
        pool_size = (
            video_cache_size
            if video_cache_size is not None
            else max(1, min(n_video_files or 1, 32))
        )
        self._video_pool = VideoReaderPool(
            self.video_backend,
            max_size=pool_size,
            video_backend_kwargs=self.video_backend_kwargs or {},
        )

    def _compute_needed_data_columns(self) -> list[str]:
        """Columns the loader actually reads from a v3.0 data parquet.

        Mirrors the keys accessed by ``_load_parquet_data`` (state/action/language
        ``original_key`` plus ``episode_index``) so projection drops nothing needed.
        """
        cols: set[str] = {"episode_index"}
        for modality_type in ("state", "action"):
            if modality_type not in self.modality_configs:
                continue
            modality_info = self.modality_meta.get(modality_type, {})
            for group_name in self.modality_configs[modality_type].modality_keys:
                if group_name in modality_info:
                    cols.add(
                        modality_info[group_name].get(
                            "original_key", DEFAULT_COLUMN_NAMES[modality_type]
                        )
                    )
        if "language" in self.modality_configs:
            for key in self.modality_configs["language"].modality_keys:
                if key in LANG_KEYS:
                    continue
                subkey = key.replace("annotation.", "")
                if subkey in self.modality_meta.get("annotation", {}):
                    cols.add(self.modality_meta["annotation"][subkey].get("original_key", key))
        return sorted(c for c in cols if isinstance(c, str))

    def _count_v30_video_files(self) -> int:
        """Number of distinct (camera, chunk, file) v3.0 mp4s the config will read."""
        if not self.video_path_pattern or "video" not in self.modality_configs:
            return 0
        files: set[tuple[str, int, int]] = set()
        for image_key in self.modality_configs["video"].modality_keys:
            meta_key = self._video_key_mapping.get(image_key, image_key)
            if meta_key not in self.modality_meta.get("video", {}):
                continue
            original_key = self.modality_meta["video"][meta_key].get(
                "original_key", f"observation.images.{meta_key}"
            )
            chunk_col = f"videos/{original_key}/chunk_index"
            file_col = f"videos/{original_key}/file_index"
            for ep in self.episodes_metadata:
                if chunk_col in ep and file_col in ep:
                    files.add((original_key, int(ep[chunk_col]), int(ep[file_col])))
        return len(files)

    def _get_data_table(self, chunk_index: int, file_index: int):
        """Return the cached column-projected Arrow table for a v3.0 data file,
        reading it from disk at most once per cache lifetime."""
        key = (chunk_index, file_index)
        cached = self._table_cache.get(key)
        if cached is not None:
            self._table_cache.move_to_end(key)
            return cached
        parquet_filename = self.data_path_pattern.format(
            chunk_index=chunk_index, file_index=file_index
        )
        parquet_path = self.dataset_path / parquet_filename
        # Project to consumed columns + memory-map so per-episode slices are zero-copy.
        table = pq.read_table(parquet_path, columns=self._data_columns, memory_map=True)
        self._table_cache[key] = table
        while len(self._table_cache) > self._table_cache_size:
            self._table_cache.popitem(last=False)  # evict least-recently-used
        return table

    def get_episode_lengths(self):
        """
        Compute original episode lengths.

        Returns:
            List of original episode lengths
        """
        episode_lengths = []
        for ep_meta in self.episodes_metadata:
            episode_lengths.append(int(ep_meta["length"]))
        return episode_lengths

    def get_episode_length(self, idx: int) -> int:
        """Get the length of a specific episode."""
        return self.episode_lengths[idx]

    def _parse_and_validate_modality_configs(
        self,
        modality_configs: dict[str, ModalityConfig],
    ) -> dict[str, ModalityConfig]:
        """
        Parse and validate modality configurations, filling in defaults where needed.

        For missing modality configs, creates default configurations:
        - video: All available camera views with single timestep
        - state: All available state keys with single timestep
        - action: All available action keys with 16-step horizon
        - language: Must be explicitly configured if needed

        Args:
            modality_configs: User-provided modality configurations

        Returns:
            Complete and validated modality configurations

        Raises:
            ValueError: If invalid modalities are specified
            AssertionError: If language modality configuration is invalid
        """
        # Filter out any modalities not handled by the dataset loader.
        unknown_modalities = [m for m in modality_configs if m not in ALLOWED_MODALITIES]
        if unknown_modalities:
            logging.debug(
                f"Skipping modalities not supported by dataset loader: {unknown_modalities}"
            )
            modality_configs = {
                k: v for k, v in modality_configs.items() if k in ALLOWED_MODALITIES
            }
        for modality in modality_configs:
            if modality == "language":
                # Language modality has special constraints.
                # Some embodiments (e.g. OXE_DROID) define multiple language keys for
                # training-time augmentation. At inference we only use the first key.
                assert len(modality_configs[modality].modality_keys) >= 1, (
                    "Language modality must have at least one key"
                )
                if len(modality_configs[modality].modality_keys) > 1:
                    logging.warning(
                        f"Language modality has {len(modality_configs[modality].modality_keys)} keys, "
                        f"only the first key will be used: {modality_configs[modality].modality_keys[0]}"
                    )
                    modality_configs[modality] = ModalityConfig(
                        delta_indices=modality_configs[modality].delta_indices,
                        modality_keys=[modality_configs[modality].modality_keys[0]],
                        sin_cos_embedding_keys=modality_configs[modality].sin_cos_embedding_keys,
                        mean_std_embedding_keys=modality_configs[modality].mean_std_embedding_keys,
                        action_configs=modality_configs[modality].action_configs[:1]
                        if modality_configs[modality].action_configs is not None
                        else None,
                    )
                assert modality_configs[modality].delta_indices == [0], (
                    "Only single timestep is supported for language modality"
                )

        # Build mapping from config video keys to dataset modality_meta video keys.
        # This handles the case where the model's pretrained config uses different
        # video key names than the dataset's modality.json (e.g., N1.6 vs N1.7 naming).
        self._video_key_mapping: dict[str, str] = {}
        if "video" in modality_configs and "video" in self.modality_meta:
            config_keys = modality_configs["video"].modality_keys
            meta_keys = list(self.modality_meta["video"].keys())
            needs_mapping = any(k not in self.modality_meta["video"] for k in config_keys)
            if needs_mapping:
                assert len(config_keys) == len(meta_keys), (
                    f"Cannot auto-map video keys: config has {len(config_keys)} keys "
                    f"{config_keys} but dataset modality meta has {len(meta_keys)} keys "
                    f"{meta_keys}. Counts must match for positional mapping."
                )
                for config_key, meta_key in zip(config_keys, meta_keys):
                    self._video_key_mapping[config_key] = meta_key
                logging.warning(
                    f"Video key mismatch between model config and dataset. "
                    f"Auto-mapping by position: {self._video_key_mapping}"
                )

        return modality_configs

    def __len__(self) -> int:
        """Return number of episodes in dataset."""
        return len(self.episodes_metadata)

    def _extract_joint_groups(
        self,
        df: pd.DataFrame,
        joint_groups: list[str],
        modality_type: str = "state",
    ) -> pd.DataFrame:
        """
        Extract specific joint groups from data arrays based on modality metadata.

        Uses the modality metadata to slice the appropriate indices from the raw data arrays,
        allowing for flexible joint group extraction (e.g., arm joints, gripper state).

        Args:
            df: DataFrame containing the raw episode data
            joint_groups: List of joint group names to extract (e.g., ["arm", "gripper"])
            modality_type: Type of modality ("state" or "action")

        Returns:
            DataFrame with columns for each requested joint group containing sliced arrays
        """
        modality_info = self.modality_meta.get(modality_type, {})
        joint_data = pd.DataFrame()

        for group_name in joint_groups:
            if group_name in modality_info:
                group_info = modality_info[group_name]
                start_idx = group_info["start"]
                end_idx = group_info["end"]
                original_key = group_info.get("original_key", DEFAULT_COLUMN_NAMES[modality_type])
                # Slice the array data for this joint group
                if isinstance(df[original_key].iloc[0], np.ndarray):
                    joint_data[group_name] = df[original_key].map(lambda x: x[start_idx:end_idx])
                else:
                    joint_data[group_name] = df[original_key]  # for strings and scalars
            else:
                print(
                    f"Warning: Joint group '{group_name}' not found in {modality_type} modality. Available groups: {list(modality_info.keys())}"
                )

        return joint_data

    def _load_parquet_data(self, episode_index: int) -> pd.DataFrame:
        """
        Load and process parquet data for a specific episode.

        Handles the complete data loading pipeline:
        1. Load raw parquet file based on chunking structure
        2. Process language annotations (convert task indices to strings)
        3. Extract state and action joint groups

        Args:
            episode_index: Index of the episode to load

        Returns:
            Processed DataFrame with all modality data
        """
        # Load raw parquet data using the version-appropriate file layout.
        if self.is_v30:
            record = self._episode_by_index[episode_index]
            chunk_index = int(record["data/chunk_index"])
            file_index = int(record["data/file_index"])
            # v3.0 packs many episodes into one parquet; read it once (cached) and
            # take this episode's contiguous row slice via its within-file offset.
            table = self._get_data_table(chunk_index, file_index)
            base = self._file_row_base[(chunk_index, file_index)]
            start = int(record["dataset_from_index"]) - base
            expected_length = int(record["dataset_to_index"]) - int(record["dataset_from_index"])

            sliced = table.slice(start, expected_length)
            if (
                sliced.num_rows != expected_length
                or not pc.all(pc.equal(sliced.column("episode_index"), episode_index)).as_py()
            ):
                # Fallback for non-contiguous/out-of-order episodes: filter the
                # already-cached table by episode_index (no extra disk read).
                sliced = table.filter(pc.equal(table.column("episode_index"), episode_index))
            original_df = sliced.to_pandas()
            assert len(original_df) == expected_length, (
                f"v3.0 episode {episode_index} slice has {len(original_df)} rows, "
                f"expected {expected_length} (from dataset_from/to_index)"
            )
        else:
            chunk_idx = episode_index // self.chunk_size
            parquet_filename = self.data_path_pattern.format(
                episode_chunk=chunk_idx, episode_index=episode_index
            )
            parquet_path = self.dataset_path / parquet_filename
            original_df = pd.read_parquet(parquet_path)
        loaded_df = pd.DataFrame()

        # Process language annotations (convert task indices to task strings)
        if "language" in self.modality_configs:
            for key in self.modality_configs["language"].modality_keys:
                # these keys will be loaded separately from episodes.jsonl
                if key in LANG_KEYS:
                    continue
                assert key.startswith("annotation.")
                subkey = key.replace("annotation.", "")
                assert subkey in self.modality_meta["annotation"], (
                    f"Key {subkey} not found in language modality"
                )
                original_key = self.modality_meta["annotation"][subkey].get("original_key", key)
                loaded_df[f"language.{key}"] = original_df[original_key].apply(
                    lambda x: self.tasks_map[x]
                )

        # Extract joint groups for state and action modalities
        for modality_type in ["state", "action"]:
            if modality_type not in self.modality_configs:
                continue
            joint_groups_df = self._extract_joint_groups(
                original_df,
                self.modality_configs[modality_type].modality_keys,
                modality_type,
            )
            for joint_group in joint_groups_df.columns:
                loaded_df[f"{modality_type}.{joint_group}"] = joint_groups_df[joint_group]

        return loaded_df

    def _load_video_data(self, episode_index: int, indices: np.ndarray) -> dict[str, np.ndarray]:
        """
        Load video data for all configured camera views at specified indices.

        Uses the configured video backend to decode video frames at the exact indices
        needed for the episode, supporting multiple camera views simultaneously.

        Args:
            episode_index: Index of the episode to load videos for
            indices: Array of indices to extract frames at

        Returns:
            Dictionary mapping camera view names to arrays of decoded frames
        """
        video_data = {}

        if not self.video_path_pattern or "video" not in self.modality_configs:
            return video_data

        chunk_idx = episode_index // self.chunk_size
        image_keys = self.modality_configs["video"].modality_keys
        record = self._episode_by_index[episode_index] if self.is_v30 else None

        for image_key in image_keys:
            # Resolve the original key used in video file naming.
            # Use the video key mapping if the config key differs from the dataset meta key.
            meta_key = self._video_key_mapping.get(image_key, image_key)
            original_key = self.modality_meta["video"][meta_key].get(
                "original_key", f"observation.images.{meta_key}"
            )
            assert original_key in self.feature_config, (
                f"Original key {original_key} not found in feature config"
            )

            if self.is_v30:
                # v3.0 concatenates many episodes into one mp4 per video key. The
                # file location and this episode's time span come from the episode
                # record; the episode's frames begin at round(from_timestamp*fps)
                # within that file, so shift the per-episode indices by that offset.
                video_filename = self.video_path_pattern.format(
                    video_key=original_key,
                    chunk_index=int(record[f"videos/{original_key}/chunk_index"]),
                    file_index=int(record[f"videos/{original_key}/file_index"]),
                )
                from_timestamp = float(record[f"videos/{original_key}/from_timestamp"])
                frame_offset = int(round(from_timestamp * self.fps))
                frame_indices = np.asarray(indices) + frame_offset
            else:
                # v2.x stores one mp4 per episode; per-episode indices are absolute.
                video_filename = self.video_path_pattern.format(
                    episode_chunk=chunk_idx,
                    video_key=original_key,
                    episode_index=episode_index,
                )
                frame_indices = indices

            video_path = self.dataset_path / video_filename

            # v3.0 packs many episodes per mp4, so reuse a pooled decoder; v2.x has
            # one mp4 per episode and keeps the stateless path.
            if self.is_v30 and self._video_pool is not None:
                video_data[image_key] = self._video_pool.get_frames_by_indices(
                    str(video_path), frame_indices
                )
            else:
                video_data[image_key] = get_frames_by_indices(
                    str(video_path),
                    frame_indices,
                    video_backend=self.video_backend,
                    video_backend_kwargs=self.video_backend_kwargs or {},
                )

        return video_data

    def _load_mask_file(self, mask_path: Path, indices: np.ndarray) -> np.ndarray:
        """Load masks from npz/npy file at specified indices."""
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask file does not exist: {mask_path}")
        suffix = mask_path.suffix.lower()
        if suffix not in {".npz", ".npy"}:
            raise ValueError(f"Only .npz or .npy mask files are supported: {mask_path}")

        if suffix == ".npy":
            masks = np.load(mask_path)
        else:
            npz_data = np.load(mask_path)
            if "arr_0" in npz_data:
                masks = npz_data["arr_0"]
            elif len(npz_data.files) == 1:
                masks = npz_data[npz_data.files[0]]
            else:
                raise ValueError(f"Mask npz must contain a single array or 'arr_0': {mask_path}")

        if masks.ndim == 2:
            masks = masks[None, ...]

        return masks[indices]

    def _load_mask_data(self, episode_index: int, indices: np.ndarray) -> dict[str, np.ndarray]:
        """
        Load mask data for all configured mask views at specified indices.
        """
        mask_data = {}

        if not self.mask_path_pattern or "mask" not in self.modality_configs:
            return mask_data

        chunk_idx = episode_index // self.chunk_size
        mask_keys = self.modality_configs["mask"].modality_keys

        for mask_key in mask_keys:
            mask_meta = self.modality_meta.get("mask", {}).get(mask_key, {})
            original_key = mask_meta.get("original_key", mask_key)
            mask_filename = self.mask_path_pattern.format(
                episode_chunk=chunk_idx,
                episode_index=episode_index,
                mask_key=original_key,
                video_key=original_key,
            )
            mask_path = self.dataset_path / mask_filename
            mask_data[mask_key] = self._load_mask_file(mask_path, indices)

        return mask_data

    def get_dataset_statistics(self) -> dict[str, Any]:
        """
        Extract dataset statistics for normalization from loaded metadata.

        Constructs a nested dictionary containing statistics (mean, std, min, max, q01, q99)
        for each joint group in state and action modalities. These statistics are used
        by processors for data normalization during training.

        Returns:
            Nested dictionary: {modality: {joint_group: {stat_type: values}}}
        """
        mapping = {"state": "observation.state", "action": "action"}
        dataset_statistics = _rec_defaultdict()

        for modality in mapping.keys():  # state, action
            for joint_key in self.modality_configs[modality].modality_keys:
                # Determine which statistics key to use
                if self.modality_meta[modality][joint_key].get("original_key", None) is not None:
                    stats_key = self.modality_meta[modality][joint_key]["original_key"]
                else:
                    stats_key = mapping[modality]

                # Extract the relevant slice of statistics
                start_idx, end_idx = (
                    self.modality_meta[modality][joint_key]["start"],
                    self.modality_meta[modality][joint_key]["end"],
                )
                for stat_type in self.stats[stats_key].keys():  # mean, std, min, max, q01, q99
                    dataset_statistics[modality][joint_key][stat_type] = self.stats[stats_key][
                        stat_type
                    ][start_idx:end_idx]
        stats = _to_plain_dict(dataset_statistics)
        # Directly add relative action stats
        if "relative_action" in self.stats:
            stats["relative_action"] = self.stats["relative_action"]
        return stats

    def create_language_from_meta(
        self, episode_meta: dict, nframes: int, lang_key: str
    ) -> list[str]:
        if lang_key == "task":
            meta_language = random.choice(episode_meta["tasks"])
            new_languages = [meta_language] * nframes
        elif lang_key == "sub_task":
            action_delta_indices = self.modality_configs["action"].delta_indices
            action_horizon = max(action_delta_indices) - min(action_delta_indices) + 1
            new_languages = [[] for _ in range(nframes)]
            sub_tasks = episode_meta["sub_tasks"]
            for sub_task in sub_tasks:
                start_idx, end_idx, sub_text = (
                    sub_task["start"],
                    sub_task["end"],
                    sub_task["text"],
                )
                horizon = action_horizon // 2
                for i in range(start_idx - horizon, end_idx):
                    if i < 0:
                        continue
                    new_languages[i].append(sub_text)
            new_languages = [i if len(i) > 0 else [""] for i in new_languages]
            new_languages = [random.choice(i) for i in new_languages]
        else:
            raise ValueError(f"Language key {lang_key} not supported")
        return new_languages

    def __getitem__(self, idx: int) -> pd.DataFrame:
        """
        Load complete episode data as a processed DataFrame.

        Combines parquet data loading and video decoding to create a unified DataFrame
        containing all modality data for the episode. Video frames are converted to
        PIL Images and stored in the DataFrame.

        Args:
            idx: Episode index to load

        Returns:
            DataFrame with columns for all modalities and timestamps, with video frames
            as PIL Images ready for further processing

        Raises:
            IndexError: If episode index is out of bounds
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Episode index {idx} out of bounds")

        episode_meta = self.episodes_metadata[idx]
        episode_id = episode_meta["episode_index"]
        nominal_length = episode_meta["length"]

        # Load and parse the parquet data
        df = self._load_parquet_data(episode_id)

        if "language" in self.modality_configs:
            lang_key = self.modality_configs["language"].modality_keys[0]
            if lang_key in LANG_KEYS:
                new_languages = self.create_language_from_meta(episode_meta, len(df), lang_key)
                df["language." + lang_key] = new_languages

        # Use actual dataframe length (might be less than nominal)
        actual_length = min(len(df), nominal_length)
        df = df.iloc[:actual_length]

        # Load synchronized video data
        video_data = self._load_video_data(episode_id, np.arange(actual_length))

        # Add video frames to dataframe as PIL Images
        for key in video_data.keys():
            assert len(video_data[key]) == len(df), (
                f"Video data for {key} has length {len(video_data[key])} but dataframe has length {len(df)}"
            )
            df[f"video.{key}"] = [frame for frame in video_data[key]]

        # Load synchronized mask data
        mask_data = self._load_mask_data(episode_id, np.arange(actual_length))
        for key in mask_data.keys():
            assert len(mask_data[key]) == len(df), (
                f"Mask data for {key} has length {len(mask_data[key])} but dataframe has length {len(df)}"
            )
            df[f"mask.{key}"] = [mask for mask in mask_data[key]]

        return df

    def get_initial_actions(self):
        """
        Load initial actions for policy initialization if available.

        Returns:
            List containing initial action dictionaries, or empty list if not available
        """
        meta_dirpath = self.dataset_path / LEROBOT_META_DIR_NAME
        initial_actions_path = meta_dirpath / INITIAL_ACTIONS_FILENAME
        if initial_actions_path.exists():
            initial_actions = load_initial_actions(initial_actions_path)
            return initial_actions  # a single-element list of dict[str, dict[str, np.ndarray]]
        else:
            return []
