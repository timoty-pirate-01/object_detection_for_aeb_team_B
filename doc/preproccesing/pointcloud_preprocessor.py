"""Point cloud preprocessor for MMDetection3D models (CenterPoint, nuScenes layout).

Adapts an in-memory LiDAR point cloud (NumPy array) to the input the selected
model expects. No file I/O, no ROS: the same class can run offline on nuScenes
data and later inside a ROS 2 node.

CenterPoint (nuScenes) expects float32 points with 5 columns:
    x, y, z, intensity, dt
where `dt` is the time lag [s] of the sweep the point belongs to (0.0 for the
current key frame). Number of points N is NOT fixed for the PyTorch model, so
`max_points` / `min_points` are optional safety / latency limits.

Processing order (see `PointCloudPreprocessor.__call__`):
    1. drop NaN / inf points
    2. optionally reset the time column (raw .bin: column 4 is `ring`, not dt)
    3. optionally keep only the key frame (dt == 0)
    4. spatial filtering (point cloud range, forward FOV)
    5. optional intensity scaling
    6. column alignment (zero padding / truncation to `target_dim`)
    7. point count limits (deterministic downsampling / padding)

The input array is never modified, and the output never shares memory with it.
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

if TYPE_CHECKING:  # only needed for type hints, keeps the module importable without mmengine
    from mmengine.config import Config


class PointCloudPreprocessor:
    """Adapts a point cloud of shape (N, C>=3) to the model input (N', target_dim)."""

    def __init__(
        self,
        target_dim: int = 5,
        point_cloud_range: Optional[Sequence[float]] = None,
        forward_fov_only: bool = False,
        keyframe_only: bool = False,
        reset_timestamp_col: bool = False,
        intensity_scale: Optional[float] = None,
        default_intensity: float = 0.0,
        max_points: Optional[int] = None,
        min_points: int = 1,
        protect_keyframe: bool = True,
        seed: Optional[int] = 42,
    ) -> None:
        """
        :param target_dim: Number of columns the model expects (nuScenes: 5, KITTI: 4).
        :param point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max]. Points outside are dropped.
        :param forward_fov_only: If True, keep only points in front of the vehicle (x > 0).
        :param keyframe_only: If True, keep only points with dt == 0 (column 4). Has no effect
                              when the input has fewer than 5 columns (all points are then
                              treated as belonging to the key frame).
        :param reset_timestamp_col: If True and the input has a 5th column, set it to 0.0.
                                    Use for raw single-sweep .bin files where column 4 is the
                                    ring index, not a time lag.
        :param intensity_scale: If set, column 3 (intensity) is multiplied by this factor
                                (e.g. 255.0 for a sensor reporting 0..1 when the model was
                                trained on 0..255). None leaves intensity untouched.
        :param default_intensity: Value used for the intensity column when the input has
                                  only (x, y, z) and the column has to be created.
        :param max_points: Upper limit on the number of points (None = unlimited).
        :param min_points: Lower limit; shorter clouds are padded with zero rows (at the origin).
        :param protect_keyframe: When downsampling, keep key frame points (dt == 0) first and
                                 fill the remaining budget randomly from older sweeps.
                                 Only applies if target_dim >= 5.
        :param seed: Seed for downsampling. The generator is re-created on every call, so the
                     same input always gives the same output regardless of call order.
                     None gives non-deterministic sampling.
        """
        if target_dim < 3:
            raise ValueError(f"target_dim must be >= 3 (x, y, z), got {target_dim}")
        if min_points < 1:
            raise ValueError(f"min_points must be >= 1, got {min_points}")
        if max_points is not None:
            if max_points < 1:
                raise ValueError(f"max_points must be >= 1, got {max_points}")
            if max_points < min_points:
                raise ValueError(f"max_points ({max_points}) must be >= min_points ({min_points})")
        if intensity_scale is not None and intensity_scale <= 0:
            raise ValueError(f"intensity_scale must be > 0, got {intensity_scale}")

        if point_cloud_range is not None:
            pcr = np.asarray(point_cloud_range, dtype=np.float32)
            if pcr.shape != (6,):
                raise ValueError(
                    "point_cloud_range must have 6 values [x_min, y_min, z_min, x_max, y_max, z_max], "
                    f"got shape {pcr.shape}"
                )
            if np.any(pcr[:3] >= pcr[3:]):
                raise ValueError(f"point_cloud_range min values must be below max values, got {pcr.tolist()}")
            self.point_cloud_range: Optional[np.ndarray] = pcr
        else:
            self.point_cloud_range = None

        self.target_dim = int(target_dim)
        self.forward_fov_only = bool(forward_fov_only)
        self.keyframe_only = bool(keyframe_only)
        self.reset_timestamp_col = bool(reset_timestamp_col)
        self.intensity_scale = intensity_scale
        self.default_intensity = float(default_intensity)
        self.max_points = max_points
        self.min_points = int(min_points)
        self.protect_keyframe = bool(protect_keyframe)
        self.seed = seed

    # ------------------------------------------------------------------ config

    @classmethod
    def from_mmdet3d_config(cls, cfg: "Config", **kwargs) -> "PointCloudPreprocessor":
        """Build the preprocessor from an MMDetection3D model config (e.g. CenterPoint).

        `point_cloud_range` and `target_dim` are read from the config; any other
        constructor argument (keyframe_only, max_points, ...) can be passed as keyword.
        """
        pc_range = cfg.get("point_cloud_range", None)

        # Expected number of columns: use_dim (or load_dim) of the test pipeline's loader.
        target_dim = 5  # nuScenes default
        for transform in cfg.test_dataloader.dataset.pipeline:
            if transform.get("type") == "LoadPointsFromFile":
                use_dim = transform.get("use_dim", transform.get("load_dim", 5))
                target_dim = len(use_dim) if isinstance(use_dim, (list, tuple)) else int(use_dim)
                break

        # Models whose range starts at x >= 0 only look forward (e.g. KITTI).
        forward_only = bool(pc_range is not None and pc_range[0] >= 0.0)

        kwargs.setdefault("forward_fov_only", forward_only)
        return cls(target_dim=target_dim, point_cloud_range=pc_range, **kwargs)

    def build_inference_config(self, base_cfg: "Config") -> "Config":
        """Copy of `base_cfg` whose test pipeline accepts an already preprocessed
        in-memory array (for `init_model` / `inference_detector`)."""
        inference_cfg = deepcopy(base_cfg)
        inference_cfg.test_dataloader.dataset.pipeline = [
            dict(
                type="LoadPointsFromFile",
                coord_type="LIDAR",
                load_dim=self.target_dim,
                use_dim=self.target_dim,
            ),
            dict(type="Pack3DDetInputs", keys=["points"]),
        ]
        return inference_cfg

    # ------------------------------------------------------------------ steps

    @staticmethod
    def _drop_non_finite(points: np.ndarray) -> np.ndarray:
        """Remove points with NaN / inf in any column (real sensors publish such points)."""
        finite = np.isfinite(points).all(axis=1)
        return points if finite.all() else points[finite]

    def _reset_timestamp(self, points: np.ndarray) -> np.ndarray:
        """Set column 4 to 0.0 (on a copy, the caller's array stays untouched)."""
        if self.reset_timestamp_col and points.shape[1] >= 5:
            points = points.copy()
            points[:, 4] = 0.0
        return points

    def _filter_keyframe(self, points: np.ndarray) -> np.ndarray:
        """Temporal reduction: keep only the current key frame (dt == 0)."""
        if self.keyframe_only and points.shape[1] >= 5:
            points = points[np.isclose(points[:, 4], 0.0)]
        return points

    def _filter_spatial_range(self, points: np.ndarray) -> np.ndarray:
        """Spatial reduction: drop points outside the FOV / point cloud range."""
        if len(points) == 0:
            return points

        mask = np.ones(len(points), dtype=bool)
        if self.forward_fov_only:
            mask &= points[:, 0] > 0.0
        if self.point_cloud_range is not None:
            x_min, y_min, z_min, x_max, y_max, z_max = self.point_cloud_range
            mask &= (
                (points[:, 0] >= x_min) & (points[:, 0] <= x_max)
                & (points[:, 1] >= y_min) & (points[:, 1] <= y_max)
                & (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
            )
        return points[mask]

    def _scale_intensity(self, points: np.ndarray) -> np.ndarray:
        if self.intensity_scale is not None and points.shape[1] >= 4:
            points = points.copy()
            points[:, 3] *= np.float32(self.intensity_scale)
        return points

    def _align_feature_dimensions(self, points: np.ndarray) -> np.ndarray:
        """Column alignment: zero padding if too few columns, truncation if too many."""
        num_points, current_dim = points.shape

        if current_dim < self.target_dim:
            padded = np.zeros((num_points, self.target_dim), dtype=np.float32)
            padded[:, :current_dim] = points
            # Only the intensity column (index 3) gets a configurable default.
            if current_dim <= 3 < self.target_dim:
                padded[:, 3] = self.default_intensity
            points = padded
        elif current_dim > self.target_dim:
            points = points[:, : self.target_dim]

        return points

    def _select_indices(self, points: np.ndarray) -> np.ndarray:
        """Indices of the points to keep when the cloud exceeds `max_points`."""
        num_points = len(points)
        rng = np.random.default_rng(self.seed)  # fresh generator: deterministic per call

        if self.protect_keyframe and self.target_dim >= 5:
            is_key = np.isclose(points[:, 4], 0.0)
            key_idx = np.flatnonzero(is_key)
            old_idx = np.flatnonzero(~is_key)
            if len(key_idx) >= self.max_points:
                keep = rng.choice(key_idx, size=self.max_points, replace=False)
            else:
                extra = rng.choice(old_idx, size=self.max_points - len(key_idx), replace=False)
                keep = np.concatenate([key_idx, extra])
        else:
            keep = rng.choice(num_points, size=self.max_points, replace=False)

        return np.sort(keep)  # keep the original point order

    def _limit_point_count(self, points: np.ndarray) -> np.ndarray:
        """Point count limits: downsample above `max_points`, zero-pad below `min_points`."""
        if self.max_points is not None and len(points) > self.max_points:
            points = points[self._select_indices(points)]

        if len(points) < self.min_points:
            padding = np.zeros((self.min_points - len(points), self.target_dim), dtype=np.float32)
            points = np.vstack((points, padding)) if len(points) > 0 else padding

        return points

    # ------------------------------------------------------------------ main

    def __call__(self, points: np.ndarray) -> np.ndarray:
        """Run the full pipeline on one point cloud.

        :param points: array-like of shape (N, C) with C >= 3, columns starting x, y, z.
        :return: C-contiguous float32 array of shape (N', target_dim). Does not share
                 memory with `points`.
        """
        original = np.asarray(points)
        if original.ndim != 2 or original.shape[1] < 3:
            raise ValueError(f"Input must have shape (N, C>=3), got {original.shape}")

        out = np.asarray(original, dtype=np.float32)  # no copy if already float32

        out = self._drop_non_finite(out)
        out = self._reset_timestamp(out)
        out = self._filter_keyframe(out)
        out = self._filter_spatial_range(out)
        out = self._scale_intensity(out)
        out = self._align_feature_dimensions(out)
        out = self._limit_point_count(out)

        out = np.ascontiguousarray(out, dtype=np.float32)
        if np.may_share_memory(out, original):
            out = out.copy()  # nothing changed the data: never hand back the caller's buffer
        return out
