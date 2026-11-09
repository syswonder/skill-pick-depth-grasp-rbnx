# Description: 「多观察位扫描」的**独立验收**测试（验收方从零写，不 import 实现方测试）。
#
# 覆盖四块，都是黑盒/几何复核，不信实现方自己的断言：
#   1) observe_scan.plan_views 的**几何约束逐条从零重算**（限位、偏移、朝下、视角、离面
#      距离、aim 半径、相机/爪尖高度、旋转差/落点间距、deviation 排序、确定性、耗时）；
#   2) estimate_surface_z 对合成平面的估计精度/无效输入/单位兜底/避开画面底部手指区；
#   3) pipeline._observe 的**行为**：标称成功不扫、可重试失败才扫、看到即停、不可重试
#      立即停、关掉不扫、全失败回标称位、模块缺失/规划器炸了/配置坏都不许拖垮整轮；
#   4) run()/stages 语义、_look 的 retryable 分类（kind 与旧客户端文案兜底）、
#      atlas_bridge._last_stage 的源码语义、vision_client 的 kind 契约。
#
# 全部用假 client / 合成深度 / 打桩的视觉结果：不连 ROS、相机、视觉服务，不依赖真机路径。
# 假标定只复用 tests/fakes.py 的 FakeCalibration（解析几何假臂），接口约定见那个文件头。
import ast
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

PKG_DIR = Path(__file__).resolve().parents[1]
GRASP_DIR = PKG_DIR / "grasp_skill"
TESTS_DIR = Path(__file__).resolve().parent
for _path in (str(GRASP_DIR), str(PKG_DIR), str(TESTS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import fakes                                                   # noqa: E402
import observe_scan                                            # noqa: E402
import pipeline                                                # noqa: E402
from grasp_skill import vision_service                         # noqa: E402

SURFACE_Z = -0.05          # 合成"地面"在基座系的高度（与假臂几何配套）


# ---------------------------------------------------------------------------
# 合成深度（自己写，不用实现方的 helper）+ 假 client / 假现场
# ---------------------------------------------------------------------------

def plane_depth(camera_pose, camera_matrix, surface_z, rows=120, cols=160,
                noise_mm=0.0, seed=0, finger_rows=0.2, finger_depth_mm=None) -> np.ndarray:
    """合成一张"相机看着水平面 z=surface_z"的深度图（uint16 毫米）。

    针孔反投影，每个像素射线与平面求交，交点沿光轴 z 即深度。画面最下面 `finger_rows`
    默认写成 0（模拟夹爪手指遮挡）；给了 `finger_depth_mm` 就把那一片写成"很近的手指"
    （验证 estimate_surface_z 确实绕开画面底部，而不是把手指当支撑面）。
    """
    pose = np.asarray(camera_pose, dtype=float).reshape(4, 4)
    camera = np.asarray(camera_matrix, dtype=float).reshape(3, 3)
    u, v = np.meshgrid(np.arange(cols, dtype=float), np.arange(rows, dtype=float))
    directions = np.stack([(u - camera[0, 2]) / camera[0, 0],
                           (v - camera[1, 2]) / camera[1, 1],
                           np.ones_like(u)], axis=-1)
    base_directions = directions @ pose[:3, :3].T
    downward = base_directions[..., 2] < -1e-9
    depth_mm = np.zeros((rows, cols), dtype=float)
    depth_mm[downward] = (surface_z - pose[2, 3]) / base_directions[..., 2][downward]
    valid = downward & (depth_mm > 0.0)
    depth_mm = np.where(valid, depth_mm * 1000.0, 0.0)
    if noise_mm:
        rng = np.random.default_rng(seed)
        depth_mm = np.where(valid, depth_mm + rng.normal(0.0, noise_mm, depth_mm.shape), 0.0)
    depth = np.clip(np.round(depth_mm), 0.0, 65535.0).astype(np.uint16)
    if finger_rows:
        start = int(round(rows * (1.0 - finger_rows)))
        if finger_depth_mm is None:
            depth[start:, :] = 0
        else:
            depth[start:, :] = int(finger_depth_mm)
    return depth


class FakeClient:
    """pipeline 第 1、2 步会碰的 client 接口；关节状态随 servo 更新，相机位姿由假标定算。"""

    def __init__(self, calib, joints_deg, image_size=(120, 160)):
        self.calib = calib
        self.joints_deg = [float(v) for v in joints_deg]
        self.gripper_m = 0.102
        self.image_size = image_size
        self.gripper_commands = []

    def state(self):
        return {"joints_deg": list(self.joints_deg),
                "gripper_m": self.gripper_m,
                "camera_pose": np.asarray(self.calib.camera(self.joints_deg), dtype=float),
                "flange": np.asarray(self.calib.flange(self.joints_deg), dtype=float),
                "tip": np.zeros(3),
                "end_pose_world": np.eye(4)}

    def gripper(self, opening_m, **kwargs):
        del kwargs
        self.gripper_m = float(opening_m)
        self.gripper_commands.append(self.gripper_m)
        return self.gripper_m

    def wrist_rgb(self):
        rows, cols = self.image_size
        return np.zeros((rows, cols, 3), dtype=np.uint8)

    def wrist_depth(self):
        rows, cols = self.image_size
        return np.zeros((rows, cols), dtype=np.uint16)


def ok_look(report=None):
    """一眼看准了（_observe 只用到 ok/report；后两步用到的字段也摆齐）。"""
    return {"ok": True,
            "report": {"detector": "stub"} if report is None else report,
            "top_point": np.array([0.30, 0.0, SURFACE_Z]),
            "plane": {"point": np.array([0.30, 0.0, SURFACE_Z]),
                      "normal": np.array([0.0, 0.0, 1.0])},
            "long_axis": np.array([1.0, 0.0, 0.0]),
            "closing": np.array([0.0, 1.0, 0.0]),
            "footprint_width": 0.04,
            "camera_base": np.eye(4)}


def failure_look(retryable=True, kind="measurement",
                 why="这一眼没量出来：目标不在腕上相机的画面里"):
    """一眼没看准：retryable 决定要不要换观察位（真 _look 按 vision_client.kind 算）。"""
    return {"ok": False, "why": why, "kind": kind,
            "retryable": bool(retryable), "report": None}


def make_rig(config=None, surface_z=SURFACE_Z, scene=None):
    """搭一条只到第 2 步的假流水线：servo 把目标角写回 client 并记下每一次走位。"""
    calib = fakes.FakeCalibration()
    calib.dist_coeffs = np.zeros(5)          # 真标定有这个属性；假标定只关心位姿/内参
    calib.gripper = None                      # run() 会读；None = 用 OPENING_M 兜底
    scene = scene or {"target": {"index": 0, "name": "object_cube",
                                 "half_size": [0.02, 0.02, 0.015]},
                      "ramp": {"plane": {"point": [0.0, 0.0, surface_z],
                                         "normal": [0.0, 0.0, 1.0]}},
                      "environment": "real"}
    client = FakeClient(calib, calib.observe_joints)
    moves = []

    def servo(joints_deg):
        client.joints_deg = [float(v) for v in joints_deg]
        moves.append(list(client.joints_deg))
        return list(client.joints_deg)

    pipe = pipeline.Pipeline(client=client, calib=calib, servo=servo, scene=scene,
                             out_dir=None,                     # 不落盘：测试不碰文件
                             vision=None,                      # 这一组只替换 _look
                             observe_scan_config=config)
    return SimpleNamespace(calib=calib, client=client, scene=scene,
                           pipeline=pipe, moves=moves)


class LookScript:
    """替掉实例的 `_look`：按调用顺序回放预设结果，并照真 _look 的样子留一份现场。

    记录 label（`2-look-photo` / `2-look-scan-0` …），用来断言"到底走了哪几眼"。
    """

    def __init__(self, rig, results, surface_z=SURFACE_Z):
        self.rig = rig
        self.results = list(results)
        self.surface_z = surface_z
        self.labels = []

    def __call__(self, label):
        self.labels.append(label)
        pose = np.asarray(self.rig.calib.camera(self.rig.client.joints_deg), dtype=float)
        depth = plane_depth(pose, self.rig.calib.camera_matrix, self.surface_z)
        self.rig.pipeline._capture = {
            "rgb": np.zeros(depth.shape + (3,), dtype=np.uint8),
            "depth": depth,
            "state": {"camera_pose": pose, "gripper_m": self.rig.client.gripper_m}}
        return self.results.pop(0)


def new_result():
    return {"ok": False, "action": "grasp", "environment": None, "problems": [], "stages": {}}


# ---------------------------------------------------------------------------
# 几何复核：从零重算 plan_views 的每一条约束
# ---------------------------------------------------------------------------

def _rotation_gap_deg(a, b):
    cosine = (np.trace(np.asarray(a, dtype=float).T @ np.asarray(b, dtype=float)) - 1.0) / 2.0
    return float(np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0))))


def assert_views_satisfy_geometry(calib, nominal_joints, surface_z, plan, config=None):
    """把 plan 里的每个位姿拿回假臂上重算，逐条核对规划规则。"""
    cfg = observe_scan.merged_config(config)
    nominal = np.asarray(nominal_joints, dtype=float)
    limits = np.rad2deg(np.asarray(calib.bounds, dtype=float).reshape(-1, 2))
    assert plan["ok"] is True
    assert 1 <= len(plan["views"]) <= int(cfg["max_views"])
    deviations = []
    rotations = [np.asarray(calib.camera(nominal), dtype=float)[:3, :3]]
    aims = [np.asarray(calib.camera(nominal), dtype=float)[:3, 3]
            + (surface_z - np.asarray(calib.camera(nominal), dtype=float)[2, 3])
            / np.asarray(calib.camera(nominal), dtype=float)[2, 2]
            * np.asarray(calib.camera(nominal), dtype=float)[:3, 2]]
    for view in plan["views"]:
        joints = np.asarray(view["joints_deg"], dtype=float)
        assert joints.size == 6
        # 1) 限位留 1° 余量（输出关节角四舍五入到 1e-3°，所以留 2e-3° 的核对容差）
        assert np.all(joints >= limits[:, 0] + 1.0 - 2e-3), (joints, limits)
        assert np.all(joints <= limits[:, 1] - 1.0 + 2e-3), (joints, limits)
        deviation = float(np.max(np.abs(joints - nominal)))
        assert deviation <= cfg["max_deviation_deg"] + 1e-6
        assert view["deviation_deg"] == pytest.approx(deviation, abs=0.06)
        deviations.append(view["deviation_deg"])
        camera = np.asarray(calib.camera(joints), dtype=float)
        axis = camera[:3, 2]
        position = camera[:3, 3]
        # 2) 光轴朝下 + 视角范围
        assert axis[2] < 0.0
        view_angle = float(np.rad2deg(np.arccos(np.clip(-axis[2], -1.0, 1.0))))
        assert cfg["view_angle_deg"][0] <= view_angle <= cfg["view_angle_deg"][1]
        assert view["view_angle_deg"] == pytest.approx(view_angle, abs=0.06)
        # 3) 光轴与支撑面的交点：离面距离 / 落点水平半径
        distance = float((surface_z - position[2]) / axis[2])
        assert cfg["distance_m"][0] - 1e-6 <= distance <= cfg["distance_m"][1] + 1e-6
        assert view["distance_m"] == pytest.approx(distance, abs=1e-3)
        aim = position + distance * axis
        assert view["aim_m"] == pytest.approx([float(v) for v in aim], abs=1e-3)
        assert aim[2] == pytest.approx(surface_z, abs=2e-4)
        aim_radius = float(np.hypot(aim[0], aim[1]))
        assert cfg["aim_radius_m"][0] - 1e-6 <= aim_radius <= cfg["aim_radius_m"][1] + 1e-6
        # 4) 相机离面 / 爪尖离面
        assert position[2] - surface_z >= cfg["min_camera_height_m"] - 1e-6
        flange = np.asarray(calib.flange(joints), dtype=float)
        tip = flange[:3, 3] + flange[:3, :3] @ np.asarray(calib.tip_in_flange,
                                                         dtype=float).reshape(3)
        assert tip[2] - surface_z >= cfg["min_tip_height_m"] - 1e-6
        rotations.append(camera[:3, :3])
        aims.append(aim)
    # 5) 选出的位姿彼此（含标称位）旋转差/落点间距都要过选样闸
    for i in range(len(rotations)):
        for j in range(i + 1, len(rotations)):
            assert _rotation_gap_deg(rotations[i], rotations[j]) >= \
                cfg["min_rotation_gap_deg"] - 0.2
            assert np.linalg.norm(aims[i][:2] - aims[j][:2]) >= cfg["min_aim_gap_m"] - 1e-6
    # 6) 按 deviation 升序（小动作先走）
    assert deviations == sorted(deviations)
    return cfg


# ---------------------------------------------------------------------------
# 1. plan_views：几何约束 / 确定性 / 上限 / 无解
# ---------------------------------------------------------------------------

def test_plan_views_all_geometry_constraints_hold():
    """默认配置 + 假臂：每个规划位姿逐条重算都合格，且选出的位姿互相差异够大。"""
    calib = fakes.FakeCalibration()
    nominal = calib.observe_joints
    plan = observe_scan.plan_views(calib, nominal, SURFACE_Z)
    assert_views_satisfy_geometry(calib, nominal, SURFACE_Z, plan)
    assert len(plan["views"]) >= 3            # 假臂几何下应能选出至少 3 个
    assert plan["nominal"]["distance_m"] == pytest.approx(0.50, abs=0.02)
    assert plan["nominal"]["view_angle_deg"] == pytest.approx(0.0, abs=1.0)
    assert any("撒点" in note for note in plan["notes"])
    assert any("最终选" in note for note in plan["notes"])


def test_plan_views_is_deterministic_and_does_not_mutate_defaults():
    """同 config/seed 两次结果逐字节一致；返回值不污染 SCAN_DEFAULTS。"""
    calib = fakes.FakeCalibration()
    nominal = calib.observe_joints
    before = json.dumps(observe_scan.SCAN_DEFAULTS, sort_keys=True)
    first = observe_scan.plan_views(calib, nominal, SURFACE_Z)
    second = observe_scan.plan_views(calib, nominal, SURFACE_Z)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert json.dumps(observe_scan.SCAN_DEFAULTS, sort_keys=True) == before
    # 拿合并后的 config 再喂回去，结果也应一致（merged_config 幂等）
    third = observe_scan.plan_views(calib, nominal, SURFACE_Z, first["config"])
    assert json.dumps(third["views"], sort_keys=True) == json.dumps(first["views"], sort_keys=True)


@pytest.mark.parametrize("max_views", [1, 2])
def test_plan_views_respects_max_views(max_views):
    calib = fakes.FakeCalibration()
    plan = observe_scan.plan_views(calib, calib.observe_joints, SURFACE_Z,
                                   {"max_views": max_views})
    assert plan["ok"] is True
    assert len(plan["views"]) == max_views
    assert_views_satisfy_geometry(calib, calib.observe_joints, SURFACE_Z, plan,
                                  {"max_views": max_views})


def test_plan_views_ok_false_with_notes_when_no_candidate():
    """约束把候选全杀了：ok=False、views 空、notes 非空且说得出卡在哪。"""
    calib = fakes.FakeCalibration()
    plan = observe_scan.plan_views(calib, calib.observe_joints, SURFACE_Z,
                                   {"distance_m": [5.0, 6.0]})
    assert plan["ok"] is False
    assert plan["views"] == []
    assert plan["notes"]
    assert any("拒绝计数" in note for note in plan["notes"])
    assert any("没有值得走" in note for note in plan["notes"])


def test_plan_views_disabled_reports_why():
    calib = fakes.FakeCalibration()
    plan = observe_scan.plan_views(calib, calib.observe_joints, SURFACE_Z,
                                   {"enabled": False})
    assert plan["ok"] is False and plan["views"] == []
    assert any("enabled=false" in note for note in plan["notes"])


def test_plan_views_bad_config_is_reported_not_raised():
    """配置写坏（distance_m 只给一个数）：plan_views 自己收敛成 ok=False + notes。"""
    calib = fakes.FakeCalibration()
    with pytest.raises(ValueError):
        observe_scan.merged_config({"distance_m": 0.5})
    plan = observe_scan.plan_views(calib, calib.observe_joints, SURFACE_Z,
                                   {"distance_m": 0.5})
    assert plan["ok"] is False and plan["views"] == []
    assert any("配置不合法" in note for note in plan["notes"])


@pytest.mark.parametrize("surface_z", [None, float("nan"), float("inf")])
def test_plan_views_without_valid_surface_z_does_not_plan(surface_z):
    calib = fakes.FakeCalibration()
    plan = observe_scan.plan_views(calib, calib.observe_joints, surface_z)
    assert plan["ok"] is False and plan["views"] == []
    assert plan["notes"]


def test_plan_views_falls_back_to_joint_limits_deg():
    """旧/假 calib 没有 `.bounds` 时用 `.joint_limits_deg()`，约束照样成立。"""
    calib = fakes.FakeCalibration()
    del calib.bounds
    plan = observe_scan.plan_views(calib, calib.observe_joints, SURFACE_Z)
    assert plan["ok"] is True and len(plan["views"]) >= 3
    limits = np.asarray(calib.joint_limits_deg(), dtype=float).reshape(-1, 2)
    for view in plan["views"]:
        joints = np.asarray(view["joints_deg"], dtype=float)
        assert np.all(joints >= limits[:, 0] + 1.0 - 2e-3)
        assert np.all(joints <= limits[:, 1] - 1.0 + 2e-3)


def test_plan_views_is_fast_enough():
    """规划（不含 import）应在 2 s 内，真机补救路径不能靠等。"""
    calib = fakes.FakeCalibration()
    start = time.perf_counter()
    observe_scan.plan_views(calib, calib.observe_joints, SURFACE_Z)
    assert time.perf_counter() - start < 2.0


def test_merged_config_normalizes_and_ignores_unknown_keys():
    defaults = observe_scan.merged_config()
    assert defaults["enabled"] is True and defaults["max_views"] == 4.0
    merged = observe_scan.merged_config({"enabled": "false", "max_views": "2",
                                         "distance_m": [1.6, 0.3],
                                         "nonsense": 123})
    assert merged["enabled"] is False
    assert merged["max_views"] == 2.0
    assert merged["distance_m"] == [0.3, 1.6]        # 写反了也能用
    assert "nonsense" not in merged
    # 返回的是深拷贝：改它不许回写默认值
    merged["spread_levels_deg"][0][0] = 999.0
    assert observe_scan.SCAN_DEFAULTS["spread_levels_deg"][0][0] == 12.0
    # 单档 spread 简写
    single = observe_scan.merged_config({"spread_levels_deg": [12, 8, 8, 18, 15, 25]})
    assert single["spread_levels_deg"] == [[12.0, 8.0, 8.0, 18.0, 15.0, 25.0]]


# ---------------------------------------------------------------------------
# 2. estimate_surface_z：精度 / 无效输入 / 单位 / 避开手指
# ---------------------------------------------------------------------------

def test_estimate_surface_z_from_synthetic_plane_within_2cm():
    calib = fakes.FakeCalibration()
    camera_pose = calib.camera(calib.observe_joints)
    for surface_z in (-0.05, -0.45):
        depth = plane_depth(camera_pose, calib.camera_matrix, surface_z)
        estimate = observe_scan.estimate_surface_z(depth, camera_pose, calib.camera_matrix)
        assert estimate is not None
        assert estimate == pytest.approx(surface_z, abs=0.02)
        assert abs(estimate - surface_z) < 0.02


def test_estimate_surface_z_robust_to_depth_noise_and_finger_occlusion():
    """±5 mm 深度噪声 + 画面底部"很近的手指"都不该把估计带偏 2 cm。"""
    calib = fakes.FakeCalibration()
    camera_pose = calib.camera(calib.observe_joints)
    depth = plane_depth(camera_pose, calib.camera_matrix, -0.45,
                        noise_mm=5.0, seed=7, finger_depth_mm=300)
    estimate = observe_scan.estimate_surface_z(depth, camera_pose, calib.camera_matrix)
    assert estimate is not None
    assert abs(estimate - (-0.45)) < 0.02


def test_estimate_surface_z_accepts_metres_by_mistake():
    """整帧都是米（中位数 < 20）时自动 ×1000，结果仍对——单位兜底不是死代码。"""
    calib = fakes.FakeCalibration()
    camera_pose = calib.camera(calib.observe_joints)
    depth_mm = plane_depth(camera_pose, calib.camera_matrix, -0.05)
    depth_m = depth_mm.astype(np.float64) / 1000.0
    estimate = observe_scan.estimate_surface_z(depth_m, camera_pose, calib.camera_matrix)
    assert estimate is not None
    assert estimate == pytest.approx(-0.05, abs=0.02)


def test_estimate_surface_z_returns_none_on_unusable_frames():
    calib = fakes.FakeCalibration()
    camera_pose = calib.camera(calib.observe_joints)
    empty = np.zeros((120, 160), dtype=np.uint16)
    assert observe_scan.estimate_surface_z(empty, camera_pose, calib.camera_matrix) is None
    # 有效点太少
    sparse = np.zeros((120, 160), dtype=np.uint16)
    sparse[10:12, 10:12] = 500
    assert observe_scan.estimate_surface_z(sparse, camera_pose, calib.camera_matrix) is None
    # 形状 / 内参坏
    full = plane_depth(camera_pose, calib.camera_matrix, -0.05)
    assert observe_scan.estimate_surface_z(np.zeros(10), camera_pose,
                                           calib.camera_matrix) is None
    bad_k = np.array(calib.camera_matrix, dtype=float)
    bad_k[0, 0] = 0.0
    assert observe_scan.estimate_surface_z(full, camera_pose, bad_k) is None


# ---------------------------------------------------------------------------
# 3. _observe 行为：什么时候扫、怎么停、失败怎么收场
# ---------------------------------------------------------------------------

def test_observe_nominal_success_never_scans(monkeypatch):
    """标称位一眼看准：除了去拍照位那一次，臂一步都不多走。"""
    rig = make_rig()
    look_script = LookScript(rig, [ok_look()])
    monkeypatch.setattr(rig.pipeline, "_look", look_script)

    result = new_result()
    look, stage = rig.pipeline._observe(result)

    assert look["ok"] is True
    assert stage is None                                   # 不产生扫描记录
    assert look_script.labels == ["2-look-photo"]
    assert rig.moves == [rig.calib.observe_joints]
    photo = result["stages"]["photo"]
    assert photo["target_joints_deg"] == photo["actual_joints_deg"]
    assert photo["look_joints_deg"] == photo["target_joints_deg"]
    assert photo["opening_mm"] == pytest.approx(pipeline.OPENING_M * 1000, abs=0.1)
    assert result["stages"]["look_photo"] == {"detector": "stub"}


def test_observe_retryable_failure_scans_and_stops_on_win(monkeypatch):
    """可重试失败 → 真规划 → 走到第 2 个观察位看准：看到即停，记录完整。"""
    rig = make_rig()
    look_script = LookScript(rig, [failure_look(), failure_look(), ok_look()])
    monkeypatch.setattr(rig.pipeline, "_look", look_script)

    result = new_result()
    look, stage = rig.pipeline._observe(result)

    assert look["ok"] is True
    assert stage is not None
    assert look_script.labels == ["2-look-photo", "2-look-scan-0", "2-look-scan-1"]
    planned = stage["planned"]
    assert len(planned) >= 3
    assert stage["triggered"] is True
    assert stage["winning_index"] == 1
    assert stage["returned_to_nominal"] is False
    assert stage["surface_z_m"] == pytest.approx(SURFACE_Z, abs=0.02)
    assert isinstance(stage["elapsed_s"], float) and stage["elapsed_s"] >= 0.0
    # 真走了哪几个位姿：拍照位 → 规划的第 1 个 → 规划的第 2 个（没有再走第 3 个）
    assert rig.moves == [rig.calib.observe_joints,
                         planned[0]["joints_deg"], planned[1]["joints_deg"]]
    assert [attempt["ok"] for attempt in stage["attempts"]] == [False, True]
    assert stage["attempts"][0]["retryable"] is True and stage["attempts"][0]["why"]
    # 现场记录跟"最后看准的那一眼"走
    assert result["stages"]["photo"]["look_joints_deg"] == \
        [round(float(v), 2) for v in planned[1]["joints_deg"]]
    assert result["stages"]["look_photo"] == {"detector": "stub"}


def test_observe_non_retryable_failure_does_not_scan(monkeypatch):
    """连不上/服务报错：换角度也没用，一步都不多走。"""
    rig = make_rig()
    look_script = LookScript(rig, [failure_look(retryable=False, kind="transport",
                                                why="连不上视觉服务")])
    monkeypatch.setattr(rig.pipeline, "_look", look_script)

    look, stage = rig.pipeline._observe(new_result())

    assert look["ok"] is False and look["retryable"] is False
    assert stage is not None and stage["triggered"] is False
    assert stage["attempts"] == [] and stage["planned"] == []
    assert any("transport" in note for note in stage["notes"])
    assert rig.moves == [rig.calib.observe_joints]
    assert look_script.labels == ["2-look-photo"]


def test_observe_disabled_config_does_not_scan(monkeypatch):
    rig = make_rig(config={"enabled": False})
    look_script = LookScript(rig, [failure_look()])
    monkeypatch.setattr(rig.pipeline, "_look", look_script)

    look, stage = rig.pipeline._observe(new_result())

    assert look["ok"] is False
    assert stage is not None and stage["enabled"] is False
    assert stage["triggered"] is False and stage["attempts"] == []
    assert any("enabled=false" in note for note in stage["notes"])
    assert rig.moves == [rig.calib.observe_joints]


def test_observe_all_scan_views_fail_returns_to_nominal_pose(monkeypatch):
    """全看不准：先走完规划位，最后**回到标称位**，并记 returned_to_nominal。"""
    rig = make_rig()
    look_script = LookScript(rig, [failure_look()] * 8)
    monkeypatch.setattr(rig.pipeline, "_look", look_script)

    look, stage = rig.pipeline._observe(new_result())

    assert look["ok"] is False
    planned = stage["planned"]
    assert stage["attempts"] and len(stage["attempts"]) == len(planned)
    assert stage["winning_index"] is None
    assert stage["returned_to_nominal"] is True
    assert rig.moves == ([rig.calib.observe_joints]
                         + [view["joints_deg"] for view in planned]
                         + [rig.calib.observe_joints])       # 最后一步是回标称位
    assert look_script.labels == ["2-look-photo"] + \
        ["2-look-scan-%d" % i for i in range(len(planned))]


def test_observe_stops_immediately_on_non_retryable_scan_failure(monkeypatch):
    """扫描途中遇到不可重试失败：后面的观察位不白走，直接回标称位。"""
    rig = make_rig()
    look_script = LookScript(rig, [failure_look(),
                                   failure_look(retryable=False, kind="server",
                                                why="视觉服务回了 500")])
    monkeypatch.setattr(rig.pipeline, "_look", look_script)

    _look, stage = rig.pipeline._observe(new_result())

    planned = stage["planned"]
    assert len(stage["attempts"]) == 1
    assert stage["attempts"][0]["retryable"] is False
    assert stage["attempts"][0]["stopped"]
    assert rig.moves == [rig.calib.observe_joints, planned[0]["joints_deg"],
                         rig.calib.observe_joints]
    assert look_script.labels == ["2-look-photo", "2-look-scan-0"]


def test_observe_missing_module_does_not_raise(monkeypatch):
    """旧包里没有 observe_scan.py：不许炸，按"标称位没看准"处理并写清原因。"""
    monkeypatch.setattr(pipeline, "observe_scan", None)
    rig = make_rig()
    look_script = LookScript(rig, [failure_look()])
    monkeypatch.setattr(rig.pipeline, "_look", look_script)

    look, stage = rig.pipeline._observe(new_result())

    assert look["ok"] is False
    assert stage is not None and stage["enabled"] is False
    assert stage["attempts"] == []
    assert any("没有 observe_scan 模块" in note for note in stage["notes"])
    assert rig.moves == [rig.calib.observe_joints]


def test_observe_plan_views_exception_is_contained(monkeypatch):
    """规划器自己炸了：不往外抛，notes 里写清，臂不动。"""
    def boom(*args, **kwargs):
        raise RuntimeError("验收桩：规划器内部错误")

    monkeypatch.setattr(observe_scan, "plan_views", boom)
    rig = make_rig()
    look_script = LookScript(rig, [failure_look()])
    monkeypatch.setattr(rig.pipeline, "_look", look_script)

    look, stage = rig.pipeline._observe(new_result())

    assert look["ok"] is False
    assert any("规划观察位出错" in note for note in stage["notes"])
    assert any("规划器内部错误" in note for note in stage["notes"])
    assert rig.moves == [rig.calib.observe_joints]


def test_observe_bad_config_does_not_scan(monkeypatch):
    """清单里 observe_scan 写坏（distance_m 单个数）：不扫、不抛、notes 有原因。"""
    rig = make_rig(config={"distance_m": 0.5})
    look_script = LookScript(rig, [failure_look()])
    monkeypatch.setattr(rig.pipeline, "_look", look_script)

    look, stage = rig.pipeline._observe(new_result())

    assert look["ok"] is False
    assert stage is not None
    assert any("配置不合法" in note for note in stage["notes"])
    assert rig.moves == [rig.calib.observe_joints]


def test_observe_estimate_error_falls_back_to_scene_plane(monkeypatch):
    """深度估面炸了：退回 scene 里的常量平面，并在 notes 里标出来源。"""
    def boom(*args, **kwargs):
        raise ValueError("验收桩：深度图形状不对")

    monkeypatch.setattr(observe_scan, "estimate_surface_z", boom)
    seen_surface = []
    real_plan = observe_scan.plan_views

    def spy(calib, nominal, surface_z, config=None):
        seen_surface.append(surface_z)
        return real_plan(calib, nominal, surface_z, config)

    monkeypatch.setattr(observe_scan, "plan_views", spy)
    rig = make_rig(scene={"target": {"index": 0, "name": "object_cube",
                                     "half_size": [0.02, 0.02, 0.015]},
                          "ramp": {"plane": {"point": [0.0, 0.0, -0.06],
                                             "normal": [0.0, 0.0, 1.0]}},
                          "environment": "real"})
    look_script = LookScript(rig, [failure_look(), failure_look(), ok_look()],
                             surface_z=-0.05)
    monkeypatch.setattr(rig.pipeline, "_look", look_script)

    _look, stage = rig.pipeline._observe(new_result())

    assert seen_surface and seen_surface[0] == pytest.approx(-0.06)
    assert stage["surface_z_m"] == pytest.approx(-0.06)
    assert any("退回配置里的常量平面" in note for note in stage["notes"])


def test_observe_no_surface_z_anywhere_does_not_scan(monkeypatch):
    """深度估不出、scene 里也没有承托面常量：不规划、不抛，notes 写清。"""
    monkeypatch.setattr(observe_scan, "estimate_surface_z", lambda *a, **k: None)
    planned = []
    monkeypatch.setattr(observe_scan, "plan_views",
                        lambda *a, **k: planned.append(1) or {"ok": False, "views": [],
                                                              "notes": []})
    rig = make_rig(scene={"target": {"index": 0, "name": "object_cube",
                                     "half_size": [0.02, 0.02, 0.015]},
                          "environment": "real"})
    look_script = LookScript(rig, [failure_look()])
    monkeypatch.setattr(rig.pipeline, "_look", look_script)

    look, stage = rig.pipeline._observe(new_result())

    assert look["ok"] is False
    assert planned == []
    assert any("没法筛观察位" in note for note in stage["notes"])
    assert rig.moves == [rig.calib.observe_joints]


def test_observe_surface_z_is_only_a_planning_prior_not_a_measurement(monkeypatch):
    """估出来的 surface_z 只喂规划器：最后那一眼的测量结果必须原样保留（不混入）。"""
    monkeypatch.setattr(observe_scan, "estimate_surface_z", lambda *a, **k: -0.06)
    rig = make_rig()
    look_script = LookScript(rig, [failure_look(), ok_look()])
    monkeypatch.setattr(rig.pipeline, "_look", look_script)

    look, stage = rig.pipeline._observe(new_result())

    assert stage["surface_z_m"] == pytest.approx(-0.06)
    # look 是视觉服务那份测量，没被"规划先验"改写
    assert float(look["plane"]["point"][2]) == pytest.approx(SURFACE_Z)
    assert float(look["top_point"][2]) == pytest.approx(SURFACE_Z)
    assert "surface_z" not in look


# ---------------------------------------------------------------------------
# 4. run()/stages 语义
# ---------------------------------------------------------------------------

def test_run_reports_scanned_problem_and_stage(monkeypatch):
    """扫过还全失败：problem 报"拍照位 + N 个备用观察位"，stages.observe_scan 在。"""
    rig = make_rig()
    lister = LookScript(rig, [failure_look()] * 8)
    monkeypatch.setattr(rig.pipeline, "_look", lister)

    result = rig.pipeline.run()

    assert result["ok"] is False
    assert len(result["problems"]) == 1
    stage = result["stages"]["observe_scan"]
    assert result["problems"][0].startswith(
        "在拍照位和 %d 个备用观察位都没看准：" % len(stage["attempts"]))
    assert stage["winning_index"] is None and stage["returned_to_nominal"] is True
    assert "photo" in result["stages"] and "look_photo" in result["stages"]


def test_run_does_not_add_scan_stage_when_nominal_look_ok(monkeypatch):
    """标称位看准了（后续因计划不过关而停）：stages 里不该有 observe_scan。"""
    rig = make_rig()
    lister = LookScript(rig, [ok_look()])
    monkeypatch.setattr(rig.pipeline, "_look", lister)
    monkeypatch.setattr(pipeline.grasp, "plan",
                        lambda *args, **kwargs: {"ok": False, "problems": ["验收桩：计划不过"]})

    result = rig.pipeline.run()

    assert result["ok"] is False
    assert "observe_scan" not in result["stages"]
    assert result["problems"] == ["计划没通过：验收桩：计划不过"]
    assert "photo" in result["stages"] and "look_photo" in result["stages"]


def test_run_keeps_old_problem_text_when_not_scanning(monkeypatch):
    """没扫（配置关掉）：problem 保持原来的"在拍照位没看准"，不冒充扫过。"""
    rig = make_rig(config={"enabled": False})
    lister = LookScript(rig, [failure_look()])
    monkeypatch.setattr(rig.pipeline, "_look", lister)

    result = rig.pipeline.run()

    assert result["ok"] is False
    assert result["problems"] == ["在拍照位没看准：这一眼没量出来：目标不在腕上相机的画面里"]
    assert result["stages"]["observe_scan"]["attempts"] == []


def test_last_stage_source_prefers_observe_scan_over_look_photo():
    """atlas_bridge._last_stage：扫描失败要报 observe_scan（从源码里取函数独立执行）。

    atlas_bridge 依赖 robonix_api（宿主上没有），所以不 import 模块，只从源码里把
    `_last_stage` 这一个函数抠出来执行——验的是仓库里那份真代码的语义。
    """
    source = (GRASP_DIR / "atlas_bridge.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "_last_stage")
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]),
                 str(GRASP_DIR / "atlas_bridge.py"), "exec"), namespace)
    last_stage = namespace["_last_stage"]

    assert last_stage({"stages": {"photo": {}, "look_photo": {},
                                  "observe_scan": {}}}) == "observe_scan"
    assert last_stage({"stages": {"photo": {}, "look_photo": {}}}) == "look_photo"
    assert last_stage({"stages": {"photo": {}}}) == "photo"
    assert last_stage({"stages": {}}) == "unknown"


# ---------------------------------------------------------------------------
# 5. _look 的 retryable 分类（kind 契约 / 旧客户端文案兜底）
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind,expected", [
    ("measurement", True), ("transport", False), ("server", False), ("request", False)])
def test_look_retryable_by_kind(kind, expected):
    """只有 kind=measurement 值得换观察位；其它三类换角度没用。"""
    rig = make_rig()

    def locate(**kwargs):
        del kwargs
        raise vision_service.VisionError(
            "这一眼没量出来：目标不在腕上相机的画面里", kind=kind)

    rig.pipeline.vision = SimpleNamespace(locate=locate)

    look = rig.pipeline._look("2-look-photo")

    assert look["ok"] is False
    assert look["kind"] == kind
    assert look["retryable"] is expected
    assert "这一眼没量出来" in look["why"]
    # 技能端不再订阅相机：现场里只留状态与失败原因，没有原始帧。
    assert rig.pipeline._capture["state"]["camera_pose"] is not None
    assert "depth" not in rig.pipeline._capture
    assert "why" in rig.pipeline._capture


@pytest.mark.parametrize("message,expected", [
    ("这一眼没量出来：目标不在画面里", True),
    ("连不上视觉服务（10.0.0.1:50051）：connection refused", False),
])
def test_look_retryable_falls_back_to_message_when_kind_missing(message, expected):
    """没有 kind 的错误（旧客户端语义）：退回文案判断，行为不变。"""
    rig = make_rig()

    def locate(**kwargs):
        del kwargs
        raise vision_service.VisionError(message)          # kind=None

    rig.pipeline.vision = SimpleNamespace(locate=locate)

    look = rig.pipeline._look("2-look-photo")

    assert look["ok"] is False
    assert look["kind"] is None
    assert look["retryable"] is expected
    assert look["why"] == message


def test_retryable_measurement_failure_helper_contract():
    class WithKind(RuntimeError):
        def __init__(self, kind):
            super().__init__("x")
            self.kind = kind

    class Old(RuntimeError):
        pass

    for kind, expected in (("measurement", True), ("transport", False),
                           ("server", False), ("request", False)):
        assert pipeline.retryable_measurement_failure(WithKind(kind)) is expected
    assert pipeline.retryable_measurement_failure(Old("这一眼没量出来：xx")) is True
    assert pipeline.retryable_measurement_failure(Old("连不上视觉服务")) is False


# ---------------------------------------------------------------------------
# 6. vision_service 的失败分类 / result_json 解析
#
# 视觉客户端的传输层是 gRPC（视频帧在服务那边取），所以这里用假 stub 测"响应怎么变成
# 结果 / 异常"，不连网、不需要真的 codegen 也能测分类逻辑。
# ---------------------------------------------------------------------------

def test_vision_error_kind_contract():
    legacy = vision_service.VisionError("旧调用只传 message")
    assert isinstance(legacy, RuntimeError)
    assert str(legacy) == "旧调用只传 message"
    assert legacy.kind is None                    # 没给 kind = 退回文案判断
    assert vision_service.VisionError("坏参数", kind="request").kind == "request"


@pytest.mark.parametrize("message,expected", [
    ("请求非法：box 要是 4 或 5 个数的数组", "request"),
    ("服务还没初始化（没有收到 CMD_INIT）", "server"),
    ("上一次 locate 还没处理完（服务一次只处理一个请求）", "server"),
    ("没给 box 且 VLM 凭据不全", "server"),
    ("目标不在腕上相机的画面里", "measurement"),
    ("框中心那块几乎没有深度（有效 0%）", "measurement"),
])
def test_classify_failure(message, expected):
    assert vision_service.classify_failure(message) == expected


def _client_with_stub(stub):
    pytest.importorskip("grasp_geometry_pb2")      # 没跑过 codegen 的裸环境跳过这组
    client = vision_service.VisionClient("127.0.0.1:50099")
    client._stub = stub                            # 绕过建通道；只测响应处理
    return client


def _payload(**overrides):
    base = {
        "ok": True,
        "top_point": [0.0, 0.0, 0.5],
        "support": {"point": [0.0, 0.0, 0.55], "normal": [0.0, 0.0, 1.0]},
        "long_axis": [1.0, 0.0, 0.0],
        "closing": [0.0, 1.0, 0.0],
        "footprint_width": 0.04,
        "bbox": [1.0, 2.0, 3.0, 4.0, 0.0],
        "frame_id": "camera_optical_frame",
        "request_id": "abc123",
        "diagnostics": {"detect_source": "vlm"},
    }
    base.update(overrides)
    return base


def test_vision_client_parses_success_result():
    payload = _payload()
    stub = SimpleNamespace(Locate=lambda request, timeout=None: SimpleNamespace(
        success=True, message="", result_json=json.dumps(payload)))
    measured = _client_with_stub(stub).locate(target="cube")
    assert measured["top_point"] == [0.0, 0.0, 0.5]
    assert measured["support"]["normal"] == [0.0, 0.0, 1.0]
    assert measured["footprint_width"] == 0.04


def test_vision_client_success_false_is_measurement():
    stub = SimpleNamespace(Locate=lambda request, timeout=None: SimpleNamespace(
        success=False, message="目标不在腕上相机的画面里",
        result_json=json.dumps({"ok": False, "why": "目标不在腕上相机的画面里",
                                "request_id": "req-1"})))
    with pytest.raises(vision_service.VisionError) as caught:
        _client_with_stub(stub).locate(target="cube")
    assert caught.value.kind == "measurement"
    assert str(caught.value).startswith("这一眼没量出来：目标不在腕上相机的画面里")
    assert "req-1" in str(caught.value)


def test_vision_client_bad_result_json_is_server():
    stub = SimpleNamespace(Locate=lambda request, timeout=None: SimpleNamespace(
        success=True, message="", result_json="not json"))
    with pytest.raises(vision_service.VisionError) as caught:
        _client_with_stub(stub).locate(target="cube")
    assert caught.value.kind == "server"


def test_vision_client_missing_fields_is_server():
    stub = SimpleNamespace(Locate=lambda request, timeout=None: SimpleNamespace(
        success=True, message="", result_json=json.dumps({"ok": True, "top_point": [0, 0, 0]})))
    with pytest.raises(vision_service.VisionError) as caught:
        _client_with_stub(stub).locate(target="cube")
    assert caught.value.kind == "server"
    assert "缺字段" in str(caught.value)


# ---------------------------------------------------------------------------
# 7. _look 成功路径：服务的相机系结果要按当前相机位姿搬到基座系
# ---------------------------------------------------------------------------

def test_look_moves_camera_frame_result_to_base_frame():
    rig = make_rig()
    camera_pose = np.asarray(rig.calib.camera(rig.client.joints_deg), dtype=float)
    payload = {
        "ok": True,
        "top_point": [0.10, -0.02, 0.50],
        "support": {"point": [0.10, -0.02, 0.58], "normal": [0.0, 0.0, 1.0]},
        "long_axis": [1.0, 0.0, 0.0],
        "closing": [0.0, 1.0, 0.0],
        "footprint_width": 0.04,
        "bbox": [10.0, 20.0, 30.0, 40.0, 0.0],
        "request_id": "req-ok",
        "diagnostics": {"detect_source": "vlm"},
    }
    rig.pipeline.vision = SimpleNamespace(locate=lambda **kwargs: payload)

    look = rig.pipeline._look("2-look-photo")

    assert look["ok"] is True
    assert look["request_id"] == "req-ok"
    expected_top = (camera_pose @ np.r_[np.asarray(payload["top_point"]), 1.0])[:3]
    np.testing.assert_allclose(look["top_point"], expected_top, atol=1e-9)
    expected_plane = (camera_pose @ np.r_[np.asarray(payload["support"]["point"]), 1.0])[:3]
    np.testing.assert_allclose(look["plane"]["point"], expected_plane, atol=1e-9)
    expected_normal = camera_pose[:3, :3] @ np.asarray(payload["support"]["normal"])
    np.testing.assert_allclose(look["plane"]["normal"], expected_normal / np.linalg.norm(expected_normal),
                               atol=1e-9)
    # 这一眼成功 → 承托面高度记进现场，供观察位扫描当先验。
    assert rig.pipeline._capture["surface_z_m"] == pytest.approx(float(expected_plane[2]))
    assert rig.pipeline._capture["state"]["camera_pose"] is not None
