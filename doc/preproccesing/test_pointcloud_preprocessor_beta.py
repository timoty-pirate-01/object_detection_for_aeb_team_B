"""Synthetic tests for PointCloudPreprocessor. Run: python test_pointcloud_preprocessor.py
(needs only numpy; no mmdet3d, no GPU, no nuScenes data)."""

import numpy as np

from doc.preproccesing.pointcloud_preprocessor_beta import PointCloudPreprocessor

CENTERPOINT_RANGE = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
RNG = np.random.default_rng(0)


def make_cloud(n, dim=5, spread=40.0, dt_values=None):
    pts = RNG.uniform(-spread, spread, size=(n, dim)).astype(np.float32)
    pts[:, 2] = RNG.uniform(-2.0, 1.0, size=n)  # z inside the range
    if dim >= 4:
        pts[:, 3] = RNG.uniform(0, 255, size=n)
    if dim >= 5:
        pts[:, 4] = 0.0 if dt_values is None else RNG.choice(dt_values, size=n)
    return pts


def test_pad_3_to_5_columns():
    out = PointCloudPreprocessor(target_dim=5)(make_cloud(1000, dim=3))
    assert out.shape == (1000, 5) and out.dtype == np.float32
    assert np.all(out[:, 3:] == 0.0)


def test_pad_default_intensity():
    out = PointCloudPreprocessor(target_dim=5, default_intensity=7.0)(make_cloud(10, dim=3))
    assert np.all(out[:, 3] == 7.0) and np.all(out[:, 4] == 0.0)


def test_pad_4_to_5_keeps_intensity():
    cloud = make_cloud(100, dim=4)
    out = PointCloudPreprocessor(target_dim=5, default_intensity=7.0)(cloud)
    assert np.array_equal(out[:, 3], cloud[:, 3])  # default only applies to a created column
    assert np.all(out[:, 4] == 0.0)


def test_truncate_5_to_4():
    cloud = make_cloud(100, dim=5)
    out = PointCloudPreprocessor(target_dim=4)(cloud)
    assert out.shape == (100, 4) and np.array_equal(out, cloud[:, :4])
    assert out.flags["C_CONTIGUOUS"]


def test_non_finite_points_removed():
    cloud = make_cloud(50, dim=5)
    cloud[3, 0] = np.nan
    cloud[7, 2] = np.inf
    cloud[9, 3] = -np.inf
    assert len(PointCloudPreprocessor()(cloud)) == 47


def test_input_not_modified_and_no_aliasing():
    cloud = make_cloud(200, dim=5)
    cloud[:, 4] = np.arange(200) % 32  # ring index in column 4
    backup = cloud.copy()
    out = PointCloudPreprocessor(reset_timestamp_col=True, intensity_scale=2.0)(cloud)
    assert np.array_equal(cloud, backup), "caller's array was modified"
    assert np.all(out[:, 4] == 0.0)
    # nothing to change at all -> output must still not be the caller's buffer
    out2 = PointCloudPreprocessor()(cloud)
    assert not np.may_share_memory(out2, cloud)


def test_raw_bin_ring_column_not_mistaken_for_dt():
    """Order fix: reset must run before the key frame filter."""
    cloud = make_cloud(500, dim=5)
    cloud[:, 4] = np.arange(500) % 32  # ring 0..31
    out = PointCloudPreprocessor(keyframe_only=True, reset_timestamp_col=True)(cloud)
    assert len(out) == 500 and np.all(out[:, 4] == 0.0)


def test_keyframe_only_on_multisweep():
    cloud = make_cloud(1000, dim=5, dt_values=[0.0, 0.05, 0.1, 0.45])
    expected = int(np.isclose(cloud[:, 4], 0.0).sum())
    out = PointCloudPreprocessor(keyframe_only=True)(cloud)
    assert len(out) == expected and np.all(out[:, 4] == 0.0)


def test_keyframe_only_no_effect_below_5_columns():
    assert len(PointCloudPreprocessor(keyframe_only=True)(make_cloud(100, dim=4))) == 100


def test_range_filter():
    cloud = make_cloud(2000, dim=5, spread=80.0)
    out = PointCloudPreprocessor(point_cloud_range=CENTERPOINT_RANGE)(cloud)
    lo, hi = np.array(CENTERPOINT_RANGE[:3]), np.array(CENTERPOINT_RANGE[3:])
    assert 0 < len(out) < len(cloud)
    assert np.all(out[:, :3] >= lo) and np.all(out[:, :3] <= hi)


def test_forward_fov_only():
    out = PointCloudPreprocessor(forward_fov_only=True)(make_cloud(2000, dim=5))
    assert 0 < len(out) < 2000 and np.all(out[:, 0] > 0.0)


def test_kitti_like_nuscenes_cloud():
    cloud = make_cloud(5000, dim=5, spread=60.0, dt_values=[0.0, 0.1])
    prep = PointCloudPreprocessor(
        target_dim=4, point_cloud_range=[0.0, -39.68, -3.0, 69.12, 39.68, 1.0],
        forward_fov_only=True, keyframe_only=True,
    )
    out = prep(cloud)
    assert out.shape[1] == 4 and 0 < len(out) < len(cloud)


def test_downsampling_deterministic_and_order_preserved():
    cloud = make_cloud(5000, dim=5, dt_values=[0.0, 0.1])
    prep = PointCloudPreprocessor(max_points=1000)
    a, b = prep(cloud), prep(cloud)  # second call must not depend on the first
    assert a.shape == (1000, 5) and np.array_equal(a, b)
    assert np.array_equal(PointCloudPreprocessor(max_points=1000)(cloud), a)  # fresh instance too


def test_downsampling_protects_keyframe():
    cloud = make_cloud(5000, dim=5, dt_values=[0.0, 0.1, 0.2])
    n_key = int(np.isclose(cloud[:, 4], 0.0).sum())
    assert n_key < 3000
    out = PointCloudPreprocessor(max_points=3000)(cloud)
    assert int(np.isclose(out[:, 4], 0.0).sum()) == n_key  # all key frame points kept
    assert len(out) == 3000
    # without protection the key frame share is only roughly preserved
    out_free = PointCloudPreprocessor(max_points=3000, protect_keyframe=False)(cloud)
    assert len(out_free) == 3000


def test_downsampling_keyframe_exceeds_budget():
    cloud = make_cloud(5000, dim=5)  # every point is key frame
    assert len(PointCloudPreprocessor(max_points=100)(cloud)) == 100


def test_min_points_padding_and_empty_cloud():
    prep = PointCloudPreprocessor(min_points=64)
    out = prep(make_cloud(10, dim=5))
    assert out.shape == (64, 5) and np.all(out[10:] == 0.0)
    empty = PointCloudPreprocessor(point_cloud_range=[0, 0, 0, 1, 1, 1], min_points=4)(make_cloud(50, dim=5, spread=30.0))
    assert empty.shape == (4, 5)
    assert PointCloudPreprocessor()(np.empty((0, 5), dtype=np.float32)).shape == (1, 5)


def test_intensity_scale():
    cloud = make_cloud(100, dim=4)
    cloud[:, 3] = RNG.uniform(0, 1, size=100)
    out = PointCloudPreprocessor(target_dim=5, intensity_scale=255.0)(cloud)
    assert np.allclose(out[:, 3], cloud[:, 3] * 255.0, rtol=1e-5)


def test_dtype_and_list_input():
    out = PointCloudPreprocessor()(make_cloud(20, dim=5).astype(np.float64).tolist())
    assert out.dtype == np.float32 and out.flags["C_CONTIGUOUS"]


def test_validation_errors():
    bad = [
        dict(target_dim=2), dict(min_points=0), dict(max_points=0),
        dict(max_points=5, min_points=10), dict(intensity_scale=0.0),
        dict(point_cloud_range=[0, 0, 0]), dict(point_cloud_range=[5, 0, 0, 1, 1, 1]),
    ]
    for kwargs in bad:
        try:
            PointCloudPreprocessor(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"no ValueError for {kwargs}")
    for bad_input in (np.zeros((10,)), np.zeros((10, 2)), np.zeros((2, 3, 4))):
        try:
            PointCloudPreprocessor()(bad_input)
        except ValueError:
            continue
        raise AssertionError(f"no ValueError for input shape {bad_input.shape}")


# --- config helpers: minimal stand-in for mmengine.Config (attribute + dict access) ---

class FakeCfg(dict):
    """Minimal stand-in for mmengine.Config: attribute access, nested dicts stay live objects."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in list(self.items()):
            if isinstance(value, dict) and not isinstance(value, FakeCfg):
                self[key] = FakeCfg(value)

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name, value):
        self[name] = value


def make_fake_centerpoint_cfg():
    return FakeCfg(
        point_cloud_range=CENTERPOINT_RANGE,
        test_dataloader=FakeCfg(dataset=FakeCfg(pipeline=[
            dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=5, use_dim=5),
            dict(type="LoadPointsFromMultiSweeps", sweeps_num=9),
            dict(type="Pack3DDetInputs", keys=["points"]),
        ])),
    )


def test_from_config_centerpoint():
    prep = PointCloudPreprocessor.from_mmdet3d_config(make_fake_centerpoint_cfg(), max_points=100_000)
    assert prep.target_dim == 5 and not prep.forward_fov_only and prep.max_points == 100_000
    assert np.allclose(prep.point_cloud_range, CENTERPOINT_RANGE)


def test_from_config_kitti_like():
    cfg = make_fake_centerpoint_cfg()
    cfg["point_cloud_range"] = [0.0, -39.68, -3.0, 69.12, 39.68, 1.0]
    cfg["test_dataloader"]["dataset"]["pipeline"][0].update(load_dim=4, use_dim=[0, 1, 2, 3])
    prep = PointCloudPreprocessor.from_mmdet3d_config(cfg)
    assert prep.target_dim == 4 and prep.forward_fov_only


def test_build_inference_config_does_not_touch_base():
    base = make_fake_centerpoint_cfg()
    inf = PointCloudPreprocessor(target_dim=5).build_inference_config(base)
    pipeline = inf["test_dataloader"]["dataset"]["pipeline"]
    assert [t["type"] for t in pipeline] == ["LoadPointsFromFile", "Pack3DDetInputs"]
    assert pipeline[0]["load_dim"] == 5 and pipeline[0]["use_dim"] == 5
    assert len(base["test_dataloader"]["dataset"]["pipeline"]) == 3


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"ok   {name}")
    print(f"\n{len(tests)} tests passed")
