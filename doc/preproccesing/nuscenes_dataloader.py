"""Single-sweep nuScenes loader producing canonical float32 [x, y, z, intensity, dt] points.

The nuScenes devkit is loaded lazily and is only needed when the caller does not
provide an already-created NuScenes instance. This module does not voxelize data
or assemble MultiSweeps.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np


PathLike = Union[str, Path]


@dataclass(frozen=True)
class NuScenesFrame:
    """A single nuScenes LIDAR_TOP keyframe and its identifying metadata."""

    sample_token: str
    lidar_sample_data_token: str
    timestamp_us: int
    lidar_path: Path
    points: np.ndarray
    sweep_count: int = 1


def _quaternion_matrix(rotation: Any) -> np.ndarray:
    """Convert a nuScenes-style [w, x, y, z] quaternion to a 3x3 matrix."""
    if len(rotation) != 4:
        raise ValueError("nuScenes rotations must contain [w, x, y, z]")

    w, x, y, z = (float(value) for value in rotation)
    norm = (w * w + x * x + y * y + z * z) ** 0.5
    if norm == 0.0:
        raise ValueError("nuScenes quaternion norm must not be zero")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm

    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _pose_matrix(record: dict[str, Any]) -> np.ndarray:
    """Create a homogeneous sensor/ego pose matrix from a nuScenes JSON record."""
    translation = record.get("translation")
    rotation = record.get("rotation")
    if not isinstance(translation, (list, tuple)) or len(translation) != 3:
        raise ValueError("nuScenes pose requires a three-value translation")

    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = _quaternion_matrix(rotation)
    matrix[:3, 3] = np.asarray(translation, dtype=np.float32)
    return matrix


def _invert_rigid_transform(matrix: np.ndarray) -> np.ndarray:
    inverse = np.eye(4, dtype=np.float32)
    inverse[:3, :3] = matrix[:3, :3].T
    inverse[:3, 3] = -(matrix[:3, :3].T @ matrix[:3, 3])
    return inverse


def _transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if points.shape[0] == 0:
        return points.copy()
    homogeneous = np.ones((points.shape[0], 4), dtype=np.float32)
    homogeneous[:, :3] = points[:, :3]
    transformed = homogeneous @ matrix.T
    return transformed[:, :3]


def _remove_close_points(points: np.ndarray, radius: float = 1.0) -> np.ndarray:
    if points.shape[0] == 0:
        return points
    close = (
        (np.abs(points[:, 0]) < radius) &
        (np.abs(points[:, 1]) < radius)
    )
    return points[~close].copy()


def load_nuscenes_lidar_bin(path: PathLike) -> np.ndarray:
    """Read one raw nuScenes LiDAR file into contiguous float32 [N, 5] points.

    Raw nuScenes rows are [x, y, z, intensity, ring]. The ring value is not the
    model's sweep-time feature, so it is dropped and the single-sweep dt is set
    to 0.0.
    """
    lidar_path = Path(path)
    if not lidar_path.is_file():
        raise FileNotFoundError(f"nuScenes LiDAR file not found: {lidar_path}")

    byte_count = lidar_path.stat().st_size
    row_size = 5 * np.dtype("<f4").itemsize
    if byte_count % row_size != 0:
        raise ValueError(
            f"nuScenes LiDAR file must contain float32 rows with 5 values: {lidar_path}"
        )

    raw_points = np.fromfile(lidar_path, dtype="<f4")
    if raw_points.size % 5 != 0:
        raise ValueError(f"nuScenes LiDAR file has an incomplete point: {lidar_path}")
    raw_points = raw_points.reshape(-1, 5)

    points = np.zeros((raw_points.shape[0], 5), dtype=np.float32)
    points[:, :4] = raw_points[:, :4]
    return points


class NuScenesDataLoader:
    """Index nuScenes samples and lazily load their single LIDAR_TOP keyframes.

    Args:
        dataroot: nuScenes data root containing the ``samples`` and
            ``sweeps`` directories.
        version: nuScenes metadata version, for example ``v1.0-mini``.
        nusc: Optional existing nuScenes devkit instance. Supplying this is
            useful when the caller already constructed the devkit or in tests.

    Items expose ``points`` in the canonical ``[x, y, z, intensity, dt]``
    layout. This loader intentionally does not assemble MultiSweeps.
    """

    def __init__(
        self,
        dataroot: PathLike,
        version: str = "v1.0-mini",
        nusc: Optional[Any] = None,
    ) -> None:
        self.dataroot = Path(dataroot)
        if not self.dataroot.is_dir():
            raise FileNotFoundError(f"nuScenes dataroot not found: {self.dataroot}")
        if not version:
            raise ValueError("nuScenes version must not be empty")

        if nusc is None:
            try:
                from nuscenes.nuscenes import NuScenes
            except ImportError as exc:
                raise ImportError(
                    "nuScenes loading requires nuscenes-devkit; install it with "
                    "`pip install nuscenes-devkit`."
                ) from exc
            nusc = NuScenes(version=version, dataroot=str(self.dataroot), verbose=False)

        self.nusc = nusc
        self._samples = self.nusc.sample

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> NuScenesFrame:
        sample = self._samples[index]
        sample_token = sample["token"]
        return self.load_sample(sample_token)

    def load_sample(self, sample_token: str) -> NuScenesFrame:
        """Load the LIDAR_TOP keyframe identified by a nuScenes sample token."""
        sample = self.nusc.get("sample", sample_token)
        data = sample.get("data", {})
        lidar_token = data.get("LIDAR_TOP")
        if not isinstance(lidar_token, str) or not lidar_token:
            raise ValueError(f"nuScenes sample has no LIDAR_TOP token: {sample_token}")

        sample_data = self.nusc.get("sample_data", lidar_token)
        filename = sample_data.get("filename")
        if not isinstance(filename, str) or not filename:
            raise ValueError(f"nuScenes LIDAR_TOP record has no filename: {lidar_token}")

        relative_path = Path(filename)
        if relative_path.is_absolute():
            lidar_path = relative_path
        else:
            lidar_path = self.dataroot / relative_path

        timestamp = sample_data.get("timestamp")
        if not isinstance(timestamp, (int, np.integer)):
            raise ValueError(f"nuScenes LIDAR_TOP record has an invalid timestamp: {lidar_token}")

        return NuScenesFrame(
            sample_token=sample_token,
            lidar_sample_data_token=lidar_token,
            timestamp_us=int(timestamp),
            lidar_path=lidar_path,
            points=load_nuscenes_lidar_bin(lidar_path),
            sweep_count=1,
        )

    def _lidar_token_for_sample(self, sample_token: str) -> tuple[str, dict[str, Any]]:
        sample = self.nusc.get("sample", sample_token)
        lidar_token = sample.get("data", {}).get("LIDAR_TOP")
        if not isinstance(lidar_token, str) or not lidar_token:
            raise ValueError(f"nuScenes sample has no LIDAR_TOP token: {sample_token}")
        return lidar_token, sample

    def _sample_data_pose(
        self,
        sample_data_token: str,
    ) -> tuple[dict[str, Any], dict[str, Any], np.ndarray, np.ndarray]:
        sample_data = self.nusc.get("sample_data", sample_data_token)
        ego_pose_token = sample_data.get("ego_pose_token")
        calibrated_sensor_token = sample_data.get("calibrated_sensor_token")
        if not ego_pose_token or not calibrated_sensor_token:
            raise ValueError(
                f"nuScenes sample_data is missing pose metadata: {sample_data_token}"
            )

        ego_pose = self.nusc.get("ego_pose", ego_pose_token)
        calibrated_sensor = self.nusc.get(
            "calibrated_sensor",
            calibrated_sensor_token,
        )
        lidar_to_ego = _pose_matrix(calibrated_sensor)
        ego_to_global = _pose_matrix(ego_pose)
        return sample_data, ego_pose, lidar_to_ego, ego_to_global

    def load_multisweep(
        self,
        sample_token: str,
        sweeps_num: int = 9,
        remove_close: bool = True,
        remove_close_radius: float = 1.0,
        pad_empty_sweeps: bool = True,
    ) -> NuScenesFrame:
        """Load a keyframe and up to `sweeps_num` previous LiDAR sweeps.

        The implementation follows the nuScenes LiDAR transformation chain:
        lidar -> ego -> global -> keyframe ego -> keyframe lidar.
        """
        if sweeps_num < 0:
            raise ValueError("sweeps_num must be non-negative")

        current_token, _ = self._lidar_token_for_sample(sample_token)
        current_sd, _, _, _ = self._sample_data_pose(current_token)
        key_timestamp = current_sd.get("timestamp")
        if not isinstance(key_timestamp, (int, np.integer)):
            raise ValueError(f"Invalid keyframe timestamp: {current_token}")

        _, _, current_lidar_to_ego, current_ego_to_global = (
            self._sample_data_pose(current_token)
        )
        current_sensor_to_global = current_ego_to_global @ current_lidar_to_ego
        global_to_current_lidar = _invert_rigid_transform(current_sensor_to_global)

        sweep_tokens = [current_token]
        previous_token = current_token
        for _ in range(sweeps_num):
            previous_sd = self.nusc.get("sample_data", previous_token)
            previous_token = previous_sd.get("prev")
            if not previous_token:
                break
            sweep_tokens.append(previous_token)

        points_by_sweep: list[np.ndarray] = []
        for index, sweep_token in enumerate(sweep_tokens):
            sweep_sd, _, lidar_to_ego, ego_to_global = (
                self._sample_data_pose(sweep_token)
            )
            sweep_timestamp = sweep_sd.get("timestamp")
            filename = sweep_sd.get("filename")
            if not isinstance(sweep_timestamp, (int, np.integer)):
                raise ValueError(f"Invalid sweep timestamp: {sweep_token}")
            if not isinstance(filename, str) or not filename:
                raise ValueError(f"nuScenes sweep has no filename: {sweep_token}")

            lidar_path = Path(filename)
            if not lidar_path.is_absolute():
                lidar_path = self.dataroot / lidar_path

            raw = np.fromfile(lidar_path, dtype="<f4")
            if raw.size % 5 != 0:
                raise ValueError(f"nuScenes LiDAR file has an incomplete point: {lidar_path}")
            raw = raw.reshape(-1, 5)

            points = np.zeros((raw.shape[0], 5), dtype=np.float32)
            points[:, :4] = raw[:, :4]

            if index == 0:
                points[:, :3] = points[:, :3].astype(np.float32)
            else:
                sensor_to_global = ego_to_global @ lidar_to_ego
                transformed_xyz = _transform_points(
                    points[:, :3],
                    global_to_current_lidar @ sensor_to_global,
                )
                points[:, :3] = transformed_xyz
            points[:, 4] = (
                (int(key_timestamp) - int(sweep_timestamp)) / 1_000_000.0
            )

            if remove_close:
                points = _remove_close_points(points, remove_close_radius)
            points_by_sweep.append(points)

        if pad_empty_sweeps:
            while len(sweep_tokens) < sweeps_num + 1:
                keyframe = points_by_sweep[0].copy()
                keyframe[:, 4] = 0.0
                if remove_close:
                    keyframe = _remove_close_points(
                        keyframe,
                        remove_close_radius,
                    )
                points_by_sweep.append(keyframe)
                sweep_tokens.append(current_token)

        combined = (
            np.concatenate(points_by_sweep, axis=0)
            if points_by_sweep
            else np.empty((0, 5), dtype=np.float32)
        )

        current_filename = current_sd.get("filename")
        if not isinstance(current_filename, str):
            raise ValueError(f"Invalid keyframe filename: {current_token}")

        return NuScenesFrame(
            sample_token=sample_token,
            lidar_sample_data_token=current_token,
            timestamp_us=int(key_timestamp),
            lidar_path=(
                Path(current_filename)
                if Path(current_filename).is_absolute()
                else self.dataroot / current_filename
            ),
            points=np.ascontiguousarray(combined, dtype=np.float32),
            sweep_count=len(sweep_tokens),
        )
