#!/usr/bin/env python3
"""Deploy the B1K R1Pro ``modality.json`` into every task dataset under a root.

All BEHAVIOR-1K R1Pro tasks share one modality layout (61-dim
``observation.state``, 23-dim ``action``, fixed camera keys), so a single
template (``examples/b1k/r1pro.json``) is copied verbatim into each
``<task>/meta/modality.json``. Before copying, each dataset's ``meta/info.json``
is validated against that layout, so any task that deviates from the expected
format is reported loudly instead of being silently mis-sliced at train time.

Usage:
    python scripts/b1k/deploy_modality.py <b1k_root> [--template PATH] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any


# Raw R1Pro layout that examples/b1k/r1pro.json is designed for.
EXPECTED_STATE_DIM = 61  # observation.state, PROPRIOCEPTION_INDICES["R1Pro"]
EXPECTED_ACTION_DIM = 23  # action, ACTION_QPOS_INDICES["R1Pro"]

DEFAULT_TEMPLATE = Path(__file__).resolve().parents[2] / "examples" / "b1k" / "r1pro.json"


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _validate_template(template: dict[str, Any]) -> None:
    """Sanity-check the template against the expected R1Pro dims.

    Guards against the template and EXPECTED_* constants silently drifting apart.
    """
    for section in ("state", "action", "video", "annotation"):
        if section not in template:
            raise ValueError(f"template missing '{section}' section")

    max_state_end = max(group["end"] for group in template["state"].values())
    if max_state_end > EXPECTED_STATE_DIM:
        raise ValueError(
            f"template state slices reach {max_state_end} > EXPECTED_STATE_DIM={EXPECTED_STATE_DIM}"
        )

    spans = sorted((g["start"], g["end"]) for g in template["action"].values())
    cursor = 0
    for start, end in spans:
        if start != cursor:
            raise ValueError(f"template action slices are not contiguous at index {cursor}")
        cursor = end
    if cursor != EXPECTED_ACTION_DIM:
        raise ValueError(
            f"template action covers {cursor} dims, expected {EXPECTED_ACTION_DIM}"
        )


def _validate_dataset(info: dict[str, Any], template: dict[str, Any]) -> list[str]:
    """Return a list of format errors (empty list means the dataset is compatible)."""
    features = info.get("features", {})
    errors: list[str] = []

    for key, expected_dim in (
        ("observation.state", EXPECTED_STATE_DIM),
        ("action", EXPECTED_ACTION_DIM),
    ):
        feature = features.get(key)
        if feature is None:
            errors.append(f"missing feature '{key}'")
            continue
        if "float" not in str(feature.get("dtype", "")):
            errors.append(f"'{key}' dtype {feature.get('dtype')!r} is not float")
        if list(feature.get("shape", [])) != [expected_dim]:
            errors.append(f"'{key}' shape {feature.get('shape')} != [{expected_dim}]")

    for video_key, meta in template["video"].items():
        original_key = meta["original_key"]
        feature = features.get(original_key)
        if feature is None:
            errors.append(f"video '{video_key}' -> missing feature '{original_key}'")
        elif feature.get("dtype") != "video":
            errors.append(
                f"video '{video_key}' -> '{original_key}' dtype {feature.get('dtype')!r} != 'video'"
            )

    for ann_key, meta in template["annotation"].items():
        original_key = meta["original_key"]
        if original_key not in features:
            errors.append(f"annotation '{ann_key}' -> missing feature '{original_key}'")

    return errors


def find_datasets(root: Path) -> list[Path]:
    """Return dataset roots (dirs containing meta/info.json) under root, recursively."""
    datasets = []
    for info_path in sorted(root.rglob("info.json")):
        if info_path.parent.name == "meta":
            datasets.append(info_path.parent.parent)
    return datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        type=Path,
        help="Root dir to search for task datasets (e.g. .../2026-challenge-demos/b1k).",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=DEFAULT_TEMPLATE,
        help=f"modality.json template to deploy (default: {DEFAULT_TEMPLATE}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without writing; exits non-zero if not in sync.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    template_path = args.template.expanduser().resolve()
    if not template_path.is_file():
        print(f"error: template not found: {template_path}", file=sys.stderr)
        return 2
    template = _load_json(template_path)
    _validate_template(template)
    template_bytes = template_path.read_bytes()

    root = args.root.expanduser().resolve()
    datasets = find_datasets(root)
    if not datasets:
        print(f"error: no datasets (meta/info.json) found under {root}", file=sys.stderr)
        return 1

    written = unchanged = failed = 0
    for dataset in datasets:
        dst = dataset / "meta" / "modality.json"
        errors = _validate_dataset(_load_json(dataset / "meta" / "info.json"), template)
        if errors:
            failed += 1
            print(f"[FAIL] {dataset}")
            for error in errors:
                print(f"         - {error}")
            continue

        if dst.exists() and dst.read_bytes() == template_bytes:
            unchanged += 1
            print(f"[ok]   {dataset} (unchanged)")
        elif args.dry_run:
            written += 1
            print(f"[plan] {dataset} (would write modality.json)")
        else:
            shutil.copyfile(template_path, dst)
            written += 1
            print(f"[write] {dataset}")

    verb = "would write" if args.dry_run else "written"
    print(
        f"\nSummary: {len(datasets)} dataset(s) | {verb}: {written} | "
        f"unchanged: {unchanged} | failed: {failed}"
    )

    if failed or (args.dry_run and written):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
