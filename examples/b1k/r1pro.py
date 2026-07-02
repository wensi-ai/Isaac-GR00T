"""Modality config for the BEHAVIOR-1K (B1K) simulated R1Pro embodiment.

Maps the OmniGibson R1Pro 61-dim proprioception (``observation.state``, ordered
by ``PROPRIOCEPTION_INDICES["R1Pro"]``) and 23-dim absolute joint action
(``ACTION_QPOS_INDICES["R1Pro"]``) into GR00T modality groups. The same
start/end layout lives in ``r1pro.json``, which serves as both the dataset
``meta/modality.json`` (training) and the serving modality config consumed by
``B1KPolicyWrapper`` at eval time.

State groups (slices of the 61-dim ``observation.state``):
    base_qvel      [0:3]    base velocity
    left_arm       [3:10]   left arm joint positions (7)
    left_gripper   [24:26]  left gripper joint positions (2)
    right_arm      [28:35]  right arm joint positions (7)
    right_gripper  [49:51]  right gripper joint positions (2)
    torso          [53:57]  trunk joint positions (4)

Action groups (slices of the 23-dim ``action``, absolute joint targets):
    base           [0:3]    base velocity command
    torso          [3:7]    trunk targets, learned relative to state.torso
    left_arm       [7:14]   left arm targets, relative to state.left_arm
    left_gripper   [14:15]  left gripper command
    right_arm      [15:22]  right arm targets, relative to state.right_arm
    right_gripper  [22:23]  right gripper command

``name`` and ``observation`` are read by ``B1KPolicyWrapper`` to locate the live
OmniGibson proprio/camera obs keys (robot name ``robot_r1``).
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


b1k_r1pro_config = {
    "name": "robot_r1",
    "observation": {
        "head": "robot_r1::robot_r1:zed_link:Camera:0::rgb",
        "left_wrist": "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb",
        "right_wrist": "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb",
    },
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["head", "left_wrist", "right_wrist"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "base_qvel",
            "torso",
            "left_arm",
            "left_gripper",
            "right_arm",
            "right_gripper",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(16)),
        modality_keys=[
            "base",
            "torso",
            "left_arm",
            "left_gripper",
            "right_arm",
            "right_gripper",
        ],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key="torso",
            ),
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key="left_arm",
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                is_gripper=True,
            ),
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key="right_arm",
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                is_gripper=True,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}


register_modality_config(b1k_r1pro_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
