"""Follow-up E11: NAVSIM v1.1's TransFuser agent as Latent TransFuser, with the camera input of one branch.

addendum E11 in docs/followup_protocol.md. The model, its checkpoint loading, the stitching and every later step are the
devkit's. Two things are added:
- branch "cheap": each of CAM_L0, CAM_F0 and CAM_R0 is downsampled by 2 per side (INTER_AREA) and upsampled back
  (INTER_LINEAR) before the devkit's own stitching; branch "full" leaves the images as released.
- the LiDAR is not loaded. With `latent: true` the backbone replaces the LiDAR feature by its learned latent and reads
  only the batch size from it (transfuser_backbone.py), so the feature is built from an empty point cloud.
"""
from __future__ import annotations

import copy
from typing import List

import cv2
import numpy as np

from navsim.agents.transfuser.transfuser_agent import TransfuserAgent
from navsim.agents.transfuser.transfuser_features import TransfuserFeatureBuilder
from navsim.common.dataclasses import SensorConfig


def degrade(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    small = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


class E11FeatureBuilder(TransfuserFeatureBuilder):
    def __init__(self, config, branch: str):
        super().__init__(config)
        assert branch in ("cheap", "full"), branch
        self._branch = branch

    def get_unique_name(self) -> str:
        return f"transfuser_feature_e11_{self._branch}"

    def compute_features(self, agent_input):
        agent_input = copy.copy(agent_input)
        cams = copy.copy(agent_input.cameras[-1])
        if self._branch == "cheap":
            for name in ("cam_l0", "cam_f0", "cam_r0"):
                c = copy.copy(getattr(cams, name))
                c.image = degrade(c.image)
                setattr(cams, name, c)
        agent_input.cameras = list(agent_input.cameras[:-1]) + [cams]
        return super().compute_features(agent_input)

    def _get_lidar_feature(self, agent_input):
        agent_input = copy.copy(agent_input)
        lid = copy.copy(agent_input.lidars[-1])
        lid.lidar_pc = np.zeros((6, 0), np.float32)
        agent_input.lidars = list(agent_input.lidars[:-1]) + [lid]
        return super()._get_lidar_feature(agent_input)


class E11LTFAgent(TransfuserAgent):
    def __init__(self, config, lr: float, checkpoint_path: str | None = None, branch: str = "full"):
        super().__init__(config, lr, checkpoint_path)
        self._branch = branch

    def name(self) -> str:
        return f"E11LTFAgent_{self._branch}"

    def get_sensor_config(self) -> SensorConfig:
        cur = [3]
        return SensorConfig(cam_f0=cur, cam_l0=cur, cam_l1=False, cam_l2=False, cam_r0=cur, cam_r1=False,
                            cam_r2=False, cam_b0=False, lidar_pc=False)

    def get_feature_builders(self) -> List:
        return [E11FeatureBuilder(self._config, self._branch)]
