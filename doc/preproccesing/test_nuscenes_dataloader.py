"""Synthetic tests for the nuScenes loader; no dataset or devkit required."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np

from nuscenes_dataloader import NuScenesDataLoader, load_nuscenes_lidar_bin


class FakeNuScenes:
    def __init__(self) -> None:
        self.sample = [{"token": "sample-1"}]
        self.records = {
            ("sample", "sample-1"): {
                "token": "sample-1",
                "data": {"LIDAR_TOP": "lidar-1"},
            },
            ("sample_data", "lidar-1"): {
                "token": "lidar-1",
                "filename": "samples/LIDAR_TOP/frame.pcd.bin",
                "timestamp": 123456789,
            },
        }

    def get(self, table_name: str, token: str) -> dict[str, Any]:
        return self.records[(table_name, token)]


class NuScenesDataLoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dataroot = Path(self.temp_dir.name)
        self.lidar_path = self.dataroot / "samples" / "LIDAR_TOP" / "frame.pcd.bin"
        self.lidar_path.parent.mkdir(parents=True)
        self.raw_points = np.array(
            [[1, 2, 3, 4, 17], [-1, -2, -3, 0.5, 28]],
            dtype="<f4",
        )
        self.raw_points.tofile(self.lidar_path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_load_sample_returns_canonical_points_and_metadata(self) -> None:
        loader = NuScenesDataLoader(self.dataroot, nusc=FakeNuScenes())

        self.assertEqual(len(loader), 1)
        frame = loader[0]
        self.assertEqual(frame.sample_token, "sample-1")
        self.assertEqual(frame.lidar_sample_data_token, "lidar-1")
        self.assertEqual(frame.timestamp_us, 123456789)
        self.assertEqual(frame.lidar_path, self.lidar_path)
        self.assertEqual(frame.points.dtype, np.float32)
        self.assertTrue(frame.points.flags["C_CONTIGUOUS"])
        np.testing.assert_array_equal(
            frame.points,
            np.array(
                [[1, 2, 3, 4, 0], [-1, -2, -3, 0.5, 0]],
                dtype=np.float32,
            ),
        )

    def test_load_sample_by_token(self) -> None:
        loader = NuScenesDataLoader(self.dataroot, nusc=FakeNuScenes())
        self.assertEqual(loader.load_sample("sample-1").points.shape, (2, 5))

    def test_empty_file_returns_empty_canonical_cloud(self) -> None:
        self.lidar_path.write_bytes(b"")
        points = load_nuscenes_lidar_bin(self.lidar_path)
        self.assertEqual(points.shape, (0, 5))
        self.assertEqual(points.dtype, np.float32)

    def test_rejects_incomplete_raw_point(self) -> None:
        self.lidar_path.write_bytes(b"\x00" * 19)
        with self.assertRaisesRegex(ValueError, "5 values"):
            load_nuscenes_lidar_bin(self.lidar_path)

    def test_missing_lidar_file_is_reported(self) -> None:
        self.lidar_path.unlink()
        with self.assertRaises(FileNotFoundError):
            load_nuscenes_lidar_bin(self.lidar_path)

    def test_sample_without_lidar_top_is_reported(self) -> None:
        nusc = FakeNuScenes()
        nusc.records[("sample", "sample-1")] = {"token": "sample-1", "data": {}}
        loader = NuScenesDataLoader(self.dataroot, nusc=nusc)
        with self.assertRaisesRegex(ValueError, "no LIDAR_TOP"):
            loader.load_sample("sample-1")


if __name__ == "__main__":
    unittest.main()
