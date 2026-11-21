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
        )
