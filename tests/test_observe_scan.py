# Description: observe_scan 的单测：多观察位规划 + 支撑面高度估计。
#
# 全部用 tests/fakes.py 里的解析假臂（手写公式的"相机 + 法兰"），不依赖真机标定、
# URDF、相机、视觉服务——这个模块本来就不碰硬件，测试也不该碰。
# 跑法：cd <包目录> && python3 -m pytest tests/ -q
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # 让 `grasp_skill` 可 import
sys.path.insert(0, str(Path(__file__).resolve().parent))       # 让 `fakes` 可 import

import fakes
from grasp_skill import observe_scan

# 假臂看着的面（基座系 z）。相机标称高度 0.45 m，所以离面 0.75 m。
SURFACE_Z = -0.30

# 给假臂量身定做的一份配置：比起默认值放宽了 aim 半径上界（假臂半径固定 0.30 m），
# 其余都收紧一点让测试快。**只是测试输入**，不是要改默认值。
FAKE_SCAN_CONFIG = {
    "spread_levels_deg": [[20.0, 12.0, 10.0, 15.0, 8.0, 8.0],
                          [40.0, 25.0, 20.0, 25.0, 15.0, 15.0]],
    "samples_per_level": 150,
    "max_deviation_deg": 50.0,
    "distance_m": [0.30, 1.20],
    "view_angle_deg": [0.0, 60.0],
    "aim_radius_m": [0.05, 0.55],
    "min_camera_height_m": 0.10,
    "min_tip_height_m": 0.10,
    "min_rotation_gap_deg": 10.0,
    "min_aim_gap_m": 0.05,
    "max_views": 4,
    "seed": 0,
}

VIEW_KEYS = {"joints_deg", "aim_m", "distance_m", "view_angle_deg",
             "deviation_deg", "rotation_gap_deg", "reason"}


def _plan(overrides=None):
    """用一份新的假 calib 跑一次 plan_views。"""
    config = dict(FAKE_SCAN_CONFIG)
    config.update(overrides or {})
    calib = fakes.FakeCalibration()
    plan = observe_scan.plan_views(calib, calib.observe_joints, SURFACE_Z, config)
    return calib, plan


# ---------------------------------------------------------------------------
# merged_config
# ---------------------------------------------------------------------------

def test_merged_config_defaults_are_normalized_copies():
    config = observe_scan.merged_config()
    assert config["enabled"] is True
    # 数值统一成 float / list[float]；比较用 ==，4.0 == 4 才通得过。
    assert config["max_views"] == 4.0
    assert config["spread_levels_deg"] == [[12.0, 8.0, 8.0, 18.0, 15.0, 25.0],
                                           [30.0, 18.0, 18.0, 35.0, 30.0, 45.0]]
    # 拿到的必须是一份新拷贝：改它不能污染 SCAN_DEFAULTS（否则下一次调用就不一样了）。
    config["distance_m"][0] = 99.0
    assert observe_scan.SCAN_DEFAULTS["distance_m"] == [0.30, 1.60]
    assert observe_scan.merged_config()["distance_m"] == [0.30, 1.60]


def test_merged_config_accepts_known_keys_and_ignores_unknown():
    config = observe_scan.merged_config({
        "max_views": 2,
        "distance_m": [1.6, 0.3],          # 写反了也要能用：统一排成 [下限, 上限]
        "enabled": "false",                # 从环境变量来可能是字符串
        "别的模块的键": {"不要理我": 1},
    })
    assert config["max_views"] == 2.0
    assert config["distance_m"] == [0.3, 1.6]
    assert config["enabled"] is False
    assert "别的模块的键" not in config
    # 非字典按"全用默认"处理：扫描是补救路径，别让一份坏配置把整轮抓取搞崩。
    assert observe_scan.merged_config("坏配置") == observe_scan.merged_config()


def test_merged_config_single_spread_level_shorthand():
    config = observe_scan.merged_config({"spread_levels_deg": [1, 2, 3, 4, 5, 6]})
    assert config["spread_levels_deg"] == [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]


# ---------------------------------------------------------------------------
# plan_views：合格性 / 每条约束 / 排序 / 确定性
# ---------------------------------------------------------------------------

def test_plan_views_returns_sorted_qualified_views():
    calib, plan = _plan()
    assert plan["ok"] is True
    assert len(plan["views"]) == 4               # 假臂几何下默认配置能选满
    config = plan["config"]
    limits = np.asarray(calib.joint_limits_deg(), dtype=float)
    nominal_joints = np.asarray(calib.observe_joints, dtype=float)
    nominal_camera = calib.camera(nominal_joints)
    nominal_rotation = nominal_camera[:3, :3]

    # 小的动作先走（按 deviation 升序）。
    deviations = [view["deviation_deg"] for view in plan["views"]]
    assert deviations == sorted(deviations)

    for view in plan["views"]:
        assert set(view) == VIEW_KEYS
        assert isinstance(view["reason"], str) and view["reason"]
        joints = np.asarray(view["joints_deg"], dtype=float)
        assert joints.size == 6 and len(view["aim_m"]) == 3

        # 1) 关节在限位内，且相对标称位姿的偏移没超上限。
        assert np.all(joints >= limits[:, 0] - 1e-6)
        assert np.all(joints <= limits[:, 1] + 1e-6)
        deviation = float(np.max(np.abs(joints - nominal_joints)))
        assert deviation <= config["max_deviation_deg"] + 1e-6
        assert abs(deviation - view["deviation_deg"]) <= 0.1

        # 2)~7) 用假臂自己的正运动学独立复算，别直接信规划器给的数。
        flange = calib.flange(joints)
        camera = flange @ calib.X
        axis = camera[:3, 2]
        position = camera[:3, 3]
        assert axis[2] < 0.0
        view_angle = float(np.degrees(np.arccos(np.clip(-axis[2], -1.0, 1.0))))
        assert config["view_angle_deg"][0] <= view_angle <= config["view_angle_deg"][1]
        distance = float((SURFACE_Z - position[2]) / axis[2])
        assert config["distance_m"][0] <= distance <= config["distance_m"][1]
        aim = position + distance * axis
        assert np.allclose(aim, view["aim_m"], atol=2e-4)      # 输出四舍五入到 0.1 mm
        assert abs(distance - view["distance_m"]) <= 2e-3
        assert abs(view_angle - view["view_angle_deg"]) <= 0.1
        aim_radius = float(np.hypot(aim[0], aim[1]))
        assert config["aim_radius_m"][0] - 1e-9 <= aim_radius <= config["aim_radius_m"][1] + 1e-9
        assert position[2] - SURFACE_Z >= config["min_camera_height_m"] - 1e-9
        tip = flange[:3, 3] + flange[:3, :3] @ calib.tip_in_flange
        assert tip[2] - SURFACE_Z >= config["min_tip_height_m"] - 1e-9

        # 相机旋转跟标称差得够远（"换了个角度"就是这个意思）。
        gap = observe_scan._rotation_gap_deg(camera[:3, :3], nominal_rotation)
        assert gap >= config["min_rotation_gap_deg"] - 0.1

    # 两两之间也都够远：挑选时每个新位姿都要跟所有已选保持这个夹角。
    for first in range(len(plan["views"])):
        for second in range(first + 1, len(plan["views"])):
            rotation_a = calib.camera(plan["views"][first]["joints_deg"])[:3, :3]
            rotation_b = calib.camera(plan["views"][second]["joints_deg"])[:3, :3]
            assert observe_scan._rotation_gap_deg(rotation_a, rotation_b) >= \
                config["min_rotation_gap_deg"] - 0.1


def test_plan_views_nominal_block_shape():
    _, plan = _plan()
    assert set(plan["nominal"]) == {"aim_m", "distance_m", "view_angle_deg"}
    assert set(plan["config"]) == set(observe_scan.SCAN_DEFAULTS)
    # 标称位自己就在假臂的正下方（半径 0.30 m、离面 0.75 m、正俯视）。
    assert abs(plan["nominal"]["aim_m"][2] - SURFACE_Z) < 1e-6
    assert abs(plan["nominal"]["distance_m"] - 0.75) < 1e-3
    assert abs(plan["nominal"]["view_angle_deg"]) < 0.1


def test_plan_views_is_deterministic():
    _, first = _plan()
    _, second = _plan()
    assert first["ok"] == second["ok"]
    assert first["views"] == second["views"]
    assert first["notes"] == second["notes"]
    assert first["nominal"] == second["nominal"]


def test_plan_views_max_views_cap():
    _, plan = _plan({"max_views": 2})
    assert plan["ok"] is True
    assert len(plan["views"]) == 2
    assert any("max_views=2" in note for note in plan["notes"])

    _, plan = _plan({"max_views": 1})
    assert plan["ok"] is True
    assert len(plan["views"]) == 1


def test_plan_views_notes_report_sampling_and_rejections():
    _, plan = _plan()
    notes = "\n".join(plan["notes"])
    assert "撒点 %d 个" % (2 * 150) in notes
    assert "各档合格数" in notes
    assert "拒绝计数" in notes
    assert "最终选 4 个" in notes
    assert "选满 max_views=4" in notes
    # 支撑面高度来自调用方，notes 里要说清用的是它（不是过时的 table_height_m）。
    assert "surface_z=-0.300 m" in notes
    assert "table_height_m" in notes


# ---------------------------------------------------------------------------
# plan_views：没解/关掉/边界
# ---------------------------------------------------------------------------

def test_plan_views_ok_false_and_notes_when_no_candidate():
    # 视角范围设到 85°~90°：朝下的光轴这辈子也到不了这个视角（θ ∈ [0°, 90°））。
    _, plan = _plan({"view_angle_deg": [85.0, 90.0]})
    assert plan["ok"] is False
    assert plan["views"] == []
    assert plan["notes"]                                   # 无解时必须留话
    assert any("一个合格候选都没有" in note for note in plan["notes"])
    assert any("拒绝计数" in note and "视角越界" in note for note in plan["notes"])


def test_plan_views_ok_false_when_aim_radius_impossible():
    _, plan = _plan({"aim_radius_m": [2.0, 3.0]})
    assert plan["ok"] is False and plan["views"] == []
    assert any("aim 半径越界" in note for note in plan["notes"])


def test_plan_views_disabled_returns_empty_with_reason():
    _, plan = _plan({"enabled": False})
    assert plan["ok"] is False
    assert plan["views"] == []
    assert any("enabled=false" in note for note in plan["notes"])


def test_plan_views_accepts_negative_seed():
    # numpy 的 default_rng 不认负数种子；配置写 -1 不该把整轮抓取炸掉（取绝对值即可）。
    _, negative = _plan({"seed": -3})
    _, positive = _plan({"seed": 3})
    assert negative["views"] == positive["views"]


def test_plan_views_bad_config_returns_failure_instead_of_raising():
    # 配置写坏（distance_m 只给一个数）不能让整轮抓取崩：退回"不扫 + 说明原因"。
    calib = fakes.FakeCalibration()
    plan = observe_scan.plan_views(calib, calib.observe_joints, SURFACE_Z,
                                   {"distance_m": [1.0]})
    assert plan["ok"] is False
    assert plan["views"] == []
    assert any("配置不合法" in note for note in plan["notes"])


def test_plan_views_bad_surface_z_returns_empty_with_reason():
    calib = fakes.FakeCalibration()
    for bad in (None, "不是数"):
        plan = observe_scan.plan_views(calib, calib.observe_joints, bad, FAKE_SCAN_CONFIG)
        assert plan["ok"] is False and plan["views"] == []
        assert any("surface_z" in note for note in plan["notes"])


def test_plan_views_has_no_bounds_falls_back_to_joint_limits_deg():
    calib, plan_with_bounds = _plan()
    del calib.bounds                                       # 只剩 .joint_limits_deg()
    assert not hasattr(calib, "bounds")
    plan_without_bounds = observe_scan.plan_views(
        calib, calib.observe_joints, SURFACE_Z, FAKE_SCAN_CONFIG)
    assert plan_without_bounds["views"] == plan_with_bounds["views"]


def test_plan_views_clips_candidates_into_limits_instead_of_stepping_out():
    # 把限位收到比 1° 余量还窄，候选只能夹到"中点附近"；仍然不许输出限位外的关节角。
    calib = fakes.FakeCalibration()
    calib.set_limits_deg([(-2.0, 20.0), (-2.0, 20.0), (-2.0, 2.0), (-2.0, 2.0),
                          (-2.0, 2.0), (-2.0, 2.0)])
    plan = observe_scan.plan_views(calib, calib.observe_joints, SURFACE_Z, FAKE_SCAN_CONFIG)
    limits = np.asarray(calib.joint_limits_deg(), dtype=float)
    for view in plan["views"]:
        joints = np.asarray(view["joints_deg"], dtype=float)
        assert np.all(joints >= limits[:, 0] - 1e-6)
        assert np.all(joints <= limits[:, 1] + 1e-6)


# ---------------------------------------------------------------------------
# estimate_surface_z
# ---------------------------------------------------------------------------

def test_estimate_surface_z_horizontal_plane():
    calib = fakes.FakeCalibration()
    camera_pose = calib.camera(calib.observe_joints)
    depth = fakes.synthetic_plane_depth(camera_pose, calib.camera_matrix, SURFACE_Z,
                                        image_size=calib.image_size)
    estimated = observe_scan.estimate_surface_z(depth, camera_pose, calib.camera_matrix)
    assert estimated is not None
    assert abs(estimated - SURFACE_Z) < 0.02               # 要求：误差 < 2 cm


def test_estimate_surface_z_tilted_camera_with_noise():
    calib = fakes.FakeCalibration()
    # 方位 25°、倾角 +18°：相机是斜着看的；再加 5 mm 深度噪声（真机量级）。
    camera_pose = calib.camera([25.0, 18.0, 0.0, 0.0, 0.0, 0.0])
    depth = fakes.synthetic_plane_depth(camera_pose, calib.camera_matrix, SURFACE_Z,
                                        image_size=calib.image_size,
                                        noise_std_mm=5.0, seed=7)
    estimated = observe_scan.estimate_surface_z(depth, camera_pose, calib.camera_matrix)
    assert estimated is not None
    assert abs(estimated - SURFACE_Z) < 0.02


def test_estimate_surface_z_accepts_meters_as_well_as_millimeters():
    calib = fakes.FakeCalibration()
    camera_pose = calib.camera(calib.observe_joints)
    depth_mm = fakes.synthetic_plane_depth(camera_pose, calib.camera_matrix, SURFACE_Z,
                                           image_size=calib.image_size)
    from_mm = observe_scan.estimate_surface_z(depth_mm, camera_pose, calib.camera_matrix)
    from_m = observe_scan.estimate_surface_z(depth_mm.astype(np.float64) / 1000.0,
                                             camera_pose, calib.camera_matrix)
    assert from_m is not None and abs(from_m - from_mm) < 1e-3


def test_estimate_surface_z_returns_none_for_all_zero():
    calib = fakes.FakeCalibration()
    camera_pose = calib.camera(calib.observe_joints)
    depth = np.zeros(calib.image_size, dtype=np.uint16)
    assert observe_scan.estimate_surface_z(depth, camera_pose, calib.camera_matrix) is None


def test_estimate_surface_z_returns_none_when_too_few_valid_points():
    calib = fakes.FakeCalibration()
    camera_pose = calib.camera(calib.observe_joints)
    depth = np.zeros(calib.image_size, dtype=np.uint16)
    depth[40:45, 60:65] = 750                              # 只有 25 个有效点，远少于下限
    assert observe_scan.estimate_surface_z(depth, camera_pose, calib.camera_matrix) is None


def test_estimate_surface_z_returns_none_for_bad_inputs():
    calib = fakes.FakeCalibration()
    camera_pose = calib.camera(calib.observe_joints)
    assert observe_scan.estimate_surface_z(None, camera_pose, calib.camera_matrix) is None
    assert observe_scan.estimate_surface_z(np.zeros((0, 0), dtype=np.uint16),
                                           camera_pose, calib.camera_matrix) is None
    assert observe_scan.estimate_surface_z(np.zeros(10, dtype=np.uint16),
                                           camera_pose, calib.camera_matrix) is None


# ---------------------------------------------------------------------------
# 多锚点配置 + "只扫基座前方"约束
# ---------------------------------------------------------------------------

def test_merged_config_normalizes_anchors():
    """anchors_deg：[[6 个数], ...]，空/null 都按空处理；数值统一成 float。"""
    config = observe_scan.merged_config({"anchors_deg": [[1, 2, 3, 4, 5, 6]],
                                         "aim_forward_min_m": "0.05"})
    assert config["anchors_deg"] == [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]
    assert config["aim_forward_min_m"] == pytest.approx(0.05)
    assert observe_scan.merged_config()["anchors_deg"] == []
    assert observe_scan.merged_config({"anchors_deg": None})["anchors_deg"] == []


def test_merged_config_rejects_broken_anchors():
    """锚点是"人指定要走到哪儿"的强约束：写坏就报错，不静默少走一组。"""
    with pytest.raises(ValueError):
        observe_scan.merged_config({"anchors_deg": [[1, 2, 3]]})
    with pytest.raises(ValueError):
        observe_scan.merged_config({"anchors_deg": [[0, 0, 0, 0, 0, float("nan")]]})
    with pytest.raises(ValueError):
        observe_scan.merged_config({"anchors_deg": 5})


def test_plan_views_forward_limit_keeps_only_front_half_plane():
    """aim_forward_min_m：落点在基座后方的候选被拒，并在 notes 里给出计数。"""
    _calib, plan = _plan({"aim_forward_min_m": 0.45})
    assert any("后方" in note for note in plan["notes"])
    for view in plan["views"]:
        assert view["aim_m"][0] >= 0.45 - 1e-9


def test_plan_views_forward_limit_can_be_negative():
    """给负值 = 放开后半平面（兼容"只在标称位周围扫"的旧行为）。"""
    _calib, plan = _plan({"aim_forward_min_m": -1.0})
    assert plan["ok"] and plan["views"]
