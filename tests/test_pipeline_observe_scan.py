# Description: pipeline「多观察位扫描」的集成测试——不连 ROS / 相机 / 视觉服务。
#
# 测的是**接线和取舍**，不是规划算法本身（规划算法在 test_observe_scan.py 里）：
#   - 标称位看准了 → 一步都不扫；
#   - 标称位"这一眼没量出来" → 走到规划出来的观察位，看准即停，winning_index 对；
#   - 关掉 / 不可重试的失败 / 没有 observe_scan 模块 / 规划失败 → 不抛异常、
#     notes 写清、不多走一步；
#   - 全看不准 → 最后回标称位姿；
#   - _look 的 retryable 分类（只有 kind="measurement" 才可重试）。
#
# 假标定用 tests/fakes.py 的 FakeCalibration（planner 写的，接口见那个文件头）；
# 假 client 在本文件里（只实现第 1、2 步会碰的那几个方法）。
import json
import sys
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

import fakes                                                   # noqa: E402  planner 的假标定
import observe_scan                                            # noqa: E402  真规划器（端到端一条）
import pipeline                                                # noqa: E402
from grasp_skill import vision_service                         # noqa: E402  视觉错误类型


# ---------------------------------------------------------------------------
# 假对象：client / 一眼的结果 / 走位脚本
# ---------------------------------------------------------------------------

class FakeClient:
    """只实现 pipeline 第 1、2 步会碰的方法：state / gripper / 两路相机。

    `moves` 记下 servo 依次走了哪些关节角（本文件的主要断言对象）；
    `gripper_commands` 记下每次开到多少（米）。
    """

    def __init__(self, joints_deg=None, gripper_m=0.102, image_size=(120, 160)):
        self.joints_deg = [float(v) for v in (joints_deg or [0.0] * 6)]
        self.gripper_m = float(gripper_m)
        self.image_size = image_size
        self.moves = []
        self.gripper_commands = []

    def state(self):
        return {"joints_deg": list(self.joints_deg), "gripper_m": self.gripper_m,
                "camera_pose": np.eye(4), "flange": np.eye(4),
                "tip": np.zeros(3), "end_pose_world": np.eye(4)}

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
    """一眼看准了的最小结果（_observe 只用到 ok/report，以及后面几步才用的几何量）。"""
    return {"ok": True,
            "report": {"detector": "vlm"} if report is None else report,
            "top_point": np.array([0.30, 0.02, -0.05]),
            "plane": {"point": np.array([0.30, 0.02, -0.05]),
                      "normal": np.array([0.0, 0.0, 1.0])},
            "long_axis": np.array([1.0, 0.0, 0.0]),
            "closing": np.array([0.0, 1.0, 0.0]),
            "footprint_width": 0.04,
            "camera_base": np.eye(4)}


def failure_look(retryable=True, kind="measurement",
                 why="这一眼没量出来：目标不在画面里"):
    """一眼没看准：retryable 决定要不要换观察位（真 _look 按 vision_client 的 kind 算）。"""
    return {"ok": False, "why": why, "kind": kind, "retryable": bool(retryable),
            "report": None}


def new_result():
    """run() 里 _run_impl 刚开始时那份 result 的形状。"""
    return {"ok": False, "action": "grasp", "environment": None,
            "problems": [], "stages": {}}


def default_scene():
    """real_adapter.scene_from_config 的形状（承托面常量只当扫描的兜底先验）。"""
    return {"target": {"index": 0, "name": "object_cube",
                       "half_size": [0.02, 0.02, 0.015]},
            "ramp": {"plane": {"point": [0.0, 0.0, -0.05],
                               "normal": [0.0, 0.0, 1.0]}},
            "environment": "real"}


def zero_capture(rig):
    """假 _look 留下的"现场"（真 _look 会把 rgb/depth/state 塞进 _capture）。

    估支撑面高度要用它；不放的话 _observe 会以为"没拍到帧"而跳过扫描。
    """
    rows, cols = rig.client.image_size
    camera_pose = np.asarray(rig.calib.camera(rig.client.joints_deg), dtype=float)
    return {"rgb": np.zeros((rows, cols, 3), dtype=np.uint8),
            "depth": np.zeros((rows, cols), dtype=np.uint16),
            "state": {"camera_pose": camera_pose.tolist(), "gripper_m": 0.102}}


def plane_capture(surface_z):
    """假 _look 留下的现场：一张"相机看着 z=surface_z 平面"的合成深度图。"""
    def capture(rig):
        camera_pose = np.asarray(rig.calib.camera(rig.client.joints_deg), dtype=float)
        depth = fakes.synthetic_plane_depth(camera_pose, rig.calib.camera_matrix, surface_z)
        return {"rgb": np.zeros(depth.shape + (3,), dtype=np.uint8), "depth": depth,
                "state": {"camera_pose": camera_pose.tolist(), "gripper_m": 0.102}}
    return capture


class LookScript:
    """替掉实例的 `_look`：按调用顺序回放预设结果，并照真 _look 的样子留现场。

    记录每次的 label（`2-look-photo` / `2-look-scan-0` / `2-look-scan-1` …），断言"到底走了哪几眼"。
    """

    def __init__(self, rig, results, capture=None):
        self.rig = rig
        self.results = list(results)
        self.labels = []
        self._capture = capture or zero_capture

    def __call__(self, label):
        self.labels.append(label)
        self.rig.pipeline._capture = self._capture(self.rig)
        return self.results.pop(0)


def make_rig(config=None, calib=None, scene=None):
    """搭一条只到第 2 步就够用的假流水线；servo 会真的把"关节角"记到 client 上。"""
    calib = calib or fakes.FakeCalibration()
    if not hasattr(calib, "dist_coeffs"):        # 真标定有；假标定只关心位姿/内参
        calib.dist_coeffs = np.zeros(5)
    client = FakeClient(joints_deg=calib.observe_joints)
    scene = scene or default_scene()

    def servo(joints_deg):
        client.joints_deg = [float(v) for v in joints_deg]
        client.moves.append(list(client.joints_deg))
        return list(client.joints_deg)

    pipe = pipeline.Pipeline(client=client, calib=calib, servo=servo, scene=scene,
                             out_dir=None,             # 不落盘：测试不碰文件
                             vision=None,              # 这一组只替换 _look，不碰真视觉服务
                             observe_scan_config=config)
    return SimpleNamespace(calib=calib, client=client, scene=scene, pipeline=pipe)


def scan_views(count=3):
    """规划器返回的备用观察位（格式见 observe_scan.plan_views）。"""
    return [{"joints_deg": [10.0 * (i + 1), -5.0 * i, 0.0, 0.0, 0.0, 0.0],
             "aim_m": [0.30, 0.10 * i, -0.05],
             "distance_m": 0.52 + 0.01 * i,
             "view_angle_deg": 18.0 + i,
             "deviation_deg": 7.0 * (i + 1),
             "rotation_gap_deg": 15.0,
             "reason": "第 %d 个备用观察位" % (i + 1)}
            for i in range(count)]


class StubScan:
    """observe_scan 的假实现：记录调用、返回预设结果（算法本身不在这个文件里测）。"""

    def __init__(self, *, enabled=True, views=None, plan_ok=True, surface_z=-0.457,
                 plan_notes=None, estimate_error=None, plan_error=None,
                 views_by_call=None, surface_z_sequence=None):
        self.enabled = enabled
        self.views = list(views or [])
        # 多锚点：每次 estimate_surface_z 依次返回这里的值（用完后退回 surface_z）——
        # 用来测"每个锚点用自己那一帧重估支撑面"。
        self.surface_z_sequence = list(surface_z_sequence or [])
        # 多锚点：第 N 次 plan_views 返回第 N 组（用完退回 self.views）——pipeline 会按
        # "标称位、锚点 1、锚点 2…" 的顺序调它，每组位姿当然不一样。
        self.views_by_call = [list(group) for group in (views_by_call or [])]
        self.plan_ok = plan_ok
        self.surface_z = surface_z
        self.plan_notes = list(plan_notes or [])
        self.estimate_error = estimate_error
        self.plan_error = plan_error
        self.calls = {"merged_config": [], "estimate_surface_z": [], "plan_views": []}

    def merged_config(self, config=None):
        config = dict(config or {})
        self.calls["merged_config"].append(config)
        return {"enabled": bool(self.enabled), "max_views": 4,
                "anchors_deg": list(config.get("anchors_deg") or [])}

    def estimate_surface_z(self, depth_mm, camera_pose, camera_matrix, config=None):
        call_index = len(self.calls["estimate_surface_z"])
        self.calls["estimate_surface_z"].append(
            {"depth_shape": np.asarray(depth_mm).shape,
             "camera_pose_shape": np.asarray(camera_pose).shape})
        if self.estimate_error is not None:
            raise self.estimate_error
        if call_index < len(self.surface_z_sequence):
            return self.surface_z_sequence[call_index]
        return self.surface_z

    def plan_views(self, calib, nominal_joints_deg, surface_z, config=None):
        call_index = len(self.calls["plan_views"])
        self.calls["plan_views"].append(
            {"calib": calib,
             "nominal_joints_deg": [float(v) for v in nominal_joints_deg],
             "surface_z": float(surface_z)})
        if self.plan_error is not None:
            raise self.plan_error
        views = (self.views_by_call[call_index]
                 if call_index < len(self.views_by_call) else self.views)
        return {"ok": bool(self.plan_ok), "views": list(views),
                "notes": list(self.plan_notes), "nominal": {}, "config": {}}


# ---------------------------------------------------------------------------
# 1、2 步的取舍：什么时候扫、什么时候不扫
# ---------------------------------------------------------------------------

def test_nominal_success_does_not_scan(monkeypatch):
    """标称位就看准了：除了去拍照位那一次，臂一步都不多走。"""
    stub = StubScan(views=scan_views())
    monkeypatch.setattr(pipeline, "observe_scan", stub)
    rig = make_rig()
    look_script = LookScript(rig, [ok_look()])
    monkeypatch.setattr(rig.pipeline, "_look", look_script)

    result = new_result()
    look, stage = rig.pipeline._observe(result)

    assert look["ok"] is True
    assert stage is None                          # 没扫：连扫描记录都不产生
    assert look_script.labels == ["2-look-photo"]
    assert rig.client.moves == [rig.calib.observe_joints]      # servo 只到过拍照位
    assert stub.calls["merged_config"] == []      # 连配置都没去读
    assert stub.calls["plan_views"] == []
    photo = result["stages"]["photo"]
    assert photo["target_joints_deg"] == [round(float(v), 2) for v in rig.calib.observe_joints]
    assert photo["actual_joints_deg"] == photo["target_joints_deg"]
    assert photo["look_joints_deg"] == photo["target_joints_deg"]
    assert photo["opening_mm"] == round(pipeline.OPENING_M * 1000, 1)
    assert result["stages"]["look_photo"] == {"detector": "vlm"}


def test_disabled_config_does_not_scan(monkeypatch):
    """清单里设 enabled: false：失败也不扫，notes 写明是被关掉的。"""
    stub = StubScan(enabled=False, views=scan_views())
    monkeypatch.setattr(pipeline, "observe_scan", stub)
    rig = make_rig(config={"enabled": False})
    monkeypatch.setattr(rig.pipeline, "_look",
                        LookScript(rig, [failure_look()]))

    look, stage = rig.pipeline._observe(new_result())

    assert look["ok"] is False and look["retryable"] is True
    assert stage is not None and stage["enabled"] is False
    assert stage["triggered"] is False
    assert stage["attempts"] == [] and stage["planned"] == []
    assert stage["winning_index"] is None
    assert any("enabled=false" in note for note in stage["notes"])
    assert rig.client.moves == [rig.calib.observe_joints]      # 没去任何备用位
    assert stub.calls["plan_views"] == []
    # 配置原样交给规划器合并（pipeline 自己**不合并**默认值）
    assert stub.calls["merged_config"] == [{"enabled": False}]


def test_non_retryable_failure_does_not_scan(monkeypatch):
    """连不上/服务报错（transport/server）：换观察位没用，直接不扫。"""
    stub = StubScan(views=scan_views())
    monkeypatch.setattr(pipeline, "observe_scan", stub)
    rig = make_rig()
    monkeypatch.setattr(rig.pipeline, "_look",
                        LookScript(rig, [failure_look(retryable=False, kind="transport",
                                                      why="连不上视觉服务")]))

    look, stage = rig.pipeline._observe(new_result())

    assert look["retryable"] is False
    assert stage is not None and stage["enabled"] is True   # 开着，但这次不值得扫
    assert stage["attempts"] == []
    assert stage["triggered"] is False
    assert any("transport" in note for note in stage["notes"])
    assert stub.calls["plan_views"] == []
    assert rig.client.moves == [rig.calib.observe_joints]


def test_scan_success_stops_on_winning_view(monkeypatch):
    """标称位没量出来 → 第 2 个备用观察位看准：走到那儿就停，不再走后面的。"""
    views = scan_views(3)
    stub = StubScan(views=views, surface_z=-0.457)
    monkeypatch.setattr(pipeline, "observe_scan", stub)
    rig = make_rig()
    look_script = LookScript(rig, [failure_look(), failure_look(), ok_look()])
    monkeypatch.setattr(rig.pipeline, "_look", look_script)

    result = new_result()
    look, stage = rig.pipeline._observe(result)

    assert look["ok"] is True                      # 返回的是看准那一份
    assert look_script.labels == ["2-look-photo", "2-look-scan-0", "2-look-scan-1"]
    assert rig.client.moves == [rig.calib.observe_joints,
                                views[0]["joints_deg"], views[1]["joints_deg"]]
    assert stage["triggered"] is True
    assert stage["surface_z_m"] == pytest.approx(-0.457)
    assert stage["planned"] == views
    assert stage["winning_index"] == 1
    assert stage["returned_to_nominal"] is False
    assert [attempt["ok"] for attempt in stage["attempts"]] == [False, True]
    assert stage["attempts"][0]["retryable"] is True
    assert stage["attempts"][0]["why"]
    assert stage["attempts"][1]["deviation_deg"] == pytest.approx(views[1]["deviation_deg"], abs=0.01)
    # 现场记录跟着"最后看准的那一眼"走（不是照片位那一份）
    assert result["stages"]["photo"]["look_joints_deg"] == views[1]["joints_deg"]
    assert result["stages"]["look_photo"] == {"detector": "vlm"}


def test_scan_all_fail_returns_to_nominal_pose(monkeypatch):
    """全看不准：最后一件事是回标称位姿，且记下 returned_to_nominal。"""
    views = scan_views(3)
    stub = StubScan(views=views, plan_notes=["最终选 3 个：到 max_views 上限"])
    monkeypatch.setattr(pipeline, "observe_scan", stub)
    rig = make_rig()
    look_script = LookScript(rig, [failure_look(), failure_look(), failure_look(),
                                   failure_look()])
    monkeypatch.setattr(rig.pipeline, "_look", look_script)

    look, stage = rig.pipeline._observe(new_result())

    assert look["ok"] is False
    assert stage["winning_index"] is None
    assert stage["returned_to_nominal"] is True
    assert len(stage["attempts"]) == 3
    # 最后一步 servo 的目标就是标称拍照位（不是停在某个随手选的观察位）
    assert rig.client.moves == [rig.calib.observe_joints] + \
        [view["joints_deg"] for view in views] + [rig.calib.observe_joints]
    assert stage["notes"]                            # 规划器的 notes 原样带出来


def test_scan_stops_after_non_retryable_failure(monkeypatch):
    """扫描途中遇到不可重试的失败（连不上）：立刻停，后面的观察位不白走。"""
    views = scan_views(3)
    stub = StubScan(views=views)
    monkeypatch.setattr(pipeline, "observe_scan", stub)
    rig = make_rig()
    look_script = LookScript(rig, [failure_look(),
                                   failure_look(),
                                   failure_look(retryable=False, kind="server",
                                                why="视觉服务回了 500")])
    monkeypatch.setattr(rig.pipeline, "_look", look_script)

    look, stage = rig.pipeline._observe(new_result())

    assert look["ok"] is False
    assert len(stage["attempts"]) == 2               # 第 3 个位姿根本没去
    assert stage["attempts"][1]["retryable"] is False
    assert "stopped" in stage["attempts"][1]
    assert rig.client.moves == [rig.calib.observe_joints, views[0]["joints_deg"],
                                views[1]["joints_deg"], rig.calib.observe_joints]
    assert stage["winning_index"] is None
    assert stage["returned_to_nominal"] is True


def test_missing_observe_scan_module_writes_note(monkeypatch):
    """旧包里没有 observe_scan.py：不许炸，按"这一眼没看准"处理。"""
    monkeypatch.setattr(pipeline, "observe_scan", None)
    rig = make_rig()
    monkeypatch.setattr(rig.pipeline, "_look", LookScript(rig, [failure_look()]))

    look, stage = rig.pipeline._observe(new_result())

    assert look["ok"] is False
    assert stage is not None and stage["enabled"] is False
    assert stage["attempts"] == []
    assert any("没有 observe_scan 模块" in note for note in stage["notes"])
    assert rig.client.moves == [rig.calib.observe_joints]


def test_plan_failure_writes_notes_without_scanning(monkeypatch):
    """规划器说没有合格位姿（ok=False）：不扫，把它的 notes 带进结果。"""
    stub = StubScan(plan_ok=False, plan_notes=["撒点 600 个：合格 0 个",
                                               "没有满足距离/视角/高度的位姿"])
    monkeypatch.setattr(pipeline, "observe_scan", stub)
    rig = make_rig()
    monkeypatch.setattr(rig.pipeline, "_look", LookScript(rig, [failure_look()]))

    look, stage = rig.pipeline._observe(new_result())

    assert look["ok"] is False
    assert stage["triggered"] is False
    assert stage["planned"] == [] and stage["attempts"] == []
    assert stage["surface_z_m"] == pytest.approx(-0.457)
    assert "撒点 600 个：合格 0 个" in stage["notes"]
    assert rig.client.moves == [rig.calib.observe_joints]


def test_plan_ok_but_no_views_is_handled(monkeypatch):
    """规划器 ok=True 却一个位姿都没给（边界情况）：也不许抛异常。"""
    stub = StubScan(plan_ok=True, views=[])
    monkeypatch.setattr(pipeline, "observe_scan", stub)
    rig = make_rig()
    monkeypatch.setattr(rig.pipeline, "_look", LookScript(rig, [failure_look()]))

    look, stage = rig.pipeline._observe(new_result())

    assert look["ok"] is False
    assert stage["attempts"] == []
    assert any("没给出可用的备用观察位" in note for note in stage["notes"])
    assert rig.client.moves == [rig.calib.observe_joints]


# ---------------------------------------------------------------------------
# 支撑面高度先验：从深度估；估不出来退场景常量；都没有就不扫
# ---------------------------------------------------------------------------

def test_surface_z_estimated_from_depth_is_passed_to_planner(monkeypatch):
    """估出来的支撑面高度原样交给规划器（规划器不读 table_height_m）。"""
    stub = StubScan(views=scan_views(1), surface_z=-0.4123)
    monkeypatch.setattr(pipeline, "observe_scan", stub)
    rig = make_rig()
    monkeypatch.setattr(rig.pipeline, "_look",
                        LookScript(rig, [failure_look(), failure_look()]))

    _look, stage = rig.pipeline._observe(new_result())

    assert stub.calls["estimate_surface_z"][0]["depth_shape"] == (120, 160)
    assert stub.calls["plan_views"][0]["surface_z"] == pytest.approx(-0.4123)
    assert stub.calls["plan_views"][0]["nominal_joints_deg"] == \
        [float(v) for v in rig.calib.observe_joints]
    assert stage["surface_z_m"] == pytest.approx(-0.4123)


def test_surface_z_falls_back_to_scene_plane(monkeypatch):
    """深度估不出支撑面：退回 scene 里的常量平面（并在 notes 里写明来源）。"""
    stub = StubScan(views=scan_views(1), surface_z=None)
    monkeypatch.setattr(pipeline, "observe_scan", stub)
    rig = make_rig()
    monkeypatch.setattr(rig.pipeline, "_look",
                        LookScript(rig, [failure_look(), failure_look()]))

    _look, stage = rig.pipeline._observe(new_result())

    assert stub.calls["plan_views"][0]["surface_z"] == pytest.approx(-0.05)
    assert stage["surface_z_m"] == pytest.approx(-0.05)
    assert any("退回配置里的常量平面" in note for note in stage["notes"])


def test_surface_z_estimate_error_falls_back_to_scene_plane(monkeypatch):
    """估面那一步自己炸了：记 warning、退回场景常量，不拖垮整轮。"""
    stub = StubScan(views=scan_views(1), estimate_error=ValueError("深度图形状不对"))
    monkeypatch.setattr(pipeline, "observe_scan", stub)
    rig = make_rig()
    monkeypatch.setattr(rig.pipeline, "_look",
                        LookScript(rig, [failure_look(), failure_look()]))

    _look, stage = rig.pipeline._observe(new_result())

    assert stub.calls["plan_views"][0]["surface_z"] == pytest.approx(-0.05)
    assert any("退回配置里的常量平面" in note for note in stage["notes"])


def test_no_surface_z_anywhere_does_not_scan(monkeypatch):
    """深度估不出、scene 里也没有承托面常量：不扫，notes 写清。"""
    stub = StubScan(views=scan_views(1), surface_z=None)
    monkeypatch.setattr(pipeline, "observe_scan", stub)
    rig = make_rig(scene={"target": {"index": 0, "name": "object_cube",
                                     "half_size": [0.02, 0.02, 0.015]},
                          "environment": "real"})
    monkeypatch.setattr(rig.pipeline, "_look", LookScript(rig, [failure_look()]))

    look, stage = rig.pipeline._observe(new_result())

    assert look["ok"] is False
    assert stub.calls["plan_views"] == []
    assert stage["attempts"] == []
    assert any("没法筛观察位" in note for note in stage["notes"])
    assert rig.client.moves == [rig.calib.observe_joints]


# ---------------------------------------------------------------------------
# run() 里的接线：problem 文案区分"扫过"和"没扫"
# ---------------------------------------------------------------------------

def test_run_reports_scanned_problem_text(monkeypatch):
    """扫过还全失败：problem 里报出"拍照位 + N 个备用观察位"。"""
    views = scan_views(2)
    monkeypatch.setattr(pipeline, "observe_scan", StubScan(views=views))
    rig = make_rig()
    monkeypatch.setattr(rig.pipeline, "_look",
                        LookScript(rig, [failure_look(), failure_look(), failure_look()]))

    result = rig.pipeline.run()

    assert result["ok"] is False
    assert len(result["problems"]) == 1
    assert result["problems"][0].startswith("在拍照位和 2 个备用观察位都没看准：")
    assert "observe_scan" in result["stages"]
    assert result["stages"]["observe_scan"]["winning_index"] is None
    assert result["stages"]["observe_scan"]["returned_to_nominal"] is True
    assert "photo" in result["stages"] and "look_photo" in result["stages"]


def test_run_keeps_nominal_problem_text_when_not_scanning(monkeypatch):
    """没扫（配置关掉）：problem 保持原来的"在拍照位没看准"。"""
    monkeypatch.setattr(pipeline, "observe_scan", StubScan(enabled=False))
    rig = make_rig(config={"enabled": False})
    monkeypatch.setattr(rig.pipeline, "_look", LookScript(rig, [failure_look()]))

    result = rig.pipeline.run()

    assert result["ok"] is False
    assert result["problems"] == ["在拍照位没看准：" + failure_look()["why"]]
    assert result["stages"]["observe_scan"]["enabled"] is False


# ---------------------------------------------------------------------------
# _look 的 retryable 分类：只有"这一眼没量出来"可重试
# ---------------------------------------------------------------------------

class LegacyVisionError(RuntimeError):
    """旧客户端那种"没有 kind"的错误（退回文案判断）。"""


def _vision_raising(error):
    """一个只实现 locate 的假视觉客户端：每次调用都抛 error。"""
    def locate(**kwargs):
        del kwargs
        raise error
    return SimpleNamespace(locate=locate)


@pytest.mark.parametrize("kind, expected_retryable", [
    ("measurement", True), ("transport", False), ("server", False), ("request", False)])
def test_look_retryable_classification(kind, expected_retryable):
    """_look 失败时带出 kind / retryable：只有 measurement 才值得换观察位。"""
    rig = make_rig()
    rig.pipeline.vision = _vision_raising(
        vision_service.VisionError("这一眼没量出来：目标不在画面里", kind=kind))

    look = rig.pipeline._look("2-look-photo")

    assert look["ok"] is False
    assert look["retryable"] is expected_retryable
    assert look["kind"] == kind
    assert look["why"]


def test_look_without_vision_client_is_a_server_failure():
    """activate 没建 VisionClient 时不要假装"没量出来"：报 server、不重试。"""
    rig = make_rig()
    look = rig.pipeline._look("2-look-photo")
    assert look["ok"] is False
    assert look["kind"] == "server"
    assert look["retryable"] is False
    assert "视觉客户端没接上" in look["why"]


def test_retryable_helper_without_kind_uses_old_message():
    """旧客户端（没有 kind）按文案判断，两个方向都对。"""
    assert pipeline.retryable_measurement_failure(
        LegacyVisionError("这一眼没量出来：目标不在画面里")) is True
    assert pipeline.retryable_measurement_failure(
        LegacyVisionError("视觉服务回了 503")) is False


# ---------------------------------------------------------------------------
# 端到端：真 observe_scan 模块 + 假标定（只有这一条用真规划器）
# ---------------------------------------------------------------------------

def test_real_module_scan_success_end_to_end(monkeypatch, capsys):
    """用真 observe_scan 跑"标称失败 → 第 2 个观察位看准"，顺便打印 stage 样例 JSON。

    只替换 `_look`（真视觉服务不能连），规划/估面都是 planner 那份真实现。
    """
    surface_z = -0.05
    calib = fakes.FakeCalibration()
    nominal = [float(v) for v in calib.observe_joints]
    planned = observe_scan.plan_views(calib, nominal, surface_z)
    assert planned["ok"] is True
    assert len(planned["views"]) >= 2

    rig = make_rig(calib=calib)
    look_script = LookScript(rig, [failure_look(), failure_look(), ok_look()],
                             capture=plane_capture(surface_z))
    monkeypatch.setattr(rig.pipeline, "_look", look_script)
    # 真模块：monkeypatch 只是为了让测试不受外部环境影响（这里就是它本身）
    monkeypatch.setattr(pipeline, "observe_scan", observe_scan)

    look, stage = rig.pipeline._observe(new_result())

    assert look["ok"] is True
    assert stage["triggered"] is True
    assert stage["winning_index"] == 1
    assert len(stage["attempts"]) == 2
    # 支撑面高度是从合成深度里真估出来的（±1 cm）
    assert stage["surface_z_m"] == pytest.approx(surface_z, abs=0.01)
    assert [view["joints_deg"] for view in stage["planned"]] == \
        [view["joints_deg"] for view in planned["views"]]
    assert [round(float(v), 2) for v in rig.client.moves[-1]] == \
        [round(float(v), 2) for v in planned["views"][1]["joints_deg"]]
    assert stage["notes"]                     # 规划器写了为什么这么选

    text = json.dumps(stage, ensure_ascii=False, indent=2, default=pipeline._json_default)
    assert json.loads(text)["winning_index"] == 1
    with capsys.disabled():                   # 让这段 JSON 在 pytest 输出里看得见
        print("\n--- stages.observe_scan 样例（标称失败 → 扫描成功）---")
        print(text)


# ---------------------------------------------------------------------------
# atlas_bridge 的接线：observe_scan 要排在 look_photo 之后（真正停下的那一步）
# ---------------------------------------------------------------------------

class _StubSkill:
    """robonix_api.Skill 的最小替身：只让模块级装饰器跑得通。"""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __getattr__(self, name):
        if name == "mcp":
            def mcp(_capability_id):
                return lambda function: function
            return mcp
        return lambda function: function

    def run(self):                                   # pragma: no cover —— 测试不启动技能
        pass


def test_atlas_bridge_config_and_last_stage_wiring(monkeypatch):
    """atlas_bridge 的接线：

    - Config 收 observe_scan 块（坏值告警按空、不拖垮技能）；
    - _last_stage 里 observe_scan 排在 look_photo **之后**——扫描失败时它是真正停下的那步；
    - on_activate 把 config.observe_scan 传给 Pipeline（见源码断言）。

    宿主上未必有 robonix_api / grasp_dev_mcp（容器里由 rbnx codegen 生成），缺什么就用最小
    假模块补上——这里验的是接线，不是那两个模块。
    """
    robonix_api = SimpleNamespace(ATLAS=object(),
                                  Err=lambda message: ("err", message),
                                  Ok=lambda: ("ok",), Skill=_StubSkill)
    grasp_dev_mcp = SimpleNamespace(**{
        name: type(name, (), {})
        for name in ("Grasp_Request", "Grasp_Response",
                     "Observe_Request", "Observe_Response")})
    monkeypatch.setitem(sys.modules, "robonix_api", robonix_api)
    monkeypatch.setitem(sys.modules, "grasp_dev_mcp", grasp_dev_mcp)
    sys.modules.pop("grasp_skill.atlas_bridge", None)   # 确保这次用假模块重新导入
    try:
        from grasp_skill import atlas_bridge
    except ImportError as error:                        # 宿主连 ROS2 python 依赖都没有
        pytest.skip("宿主缺 ROS2 python 依赖（%s），跳过 atlas_bridge 接线检查" % error)

    # 扫描失败时 stages 里 photo/look_photo/observe_scan 都在：真正停下的是扫描
    assert atlas_bridge._last_stage(
        {"stages": {"photo": {}, "look_photo": {}, "observe_scan": {}}}) == "observe_scan"
    assert atlas_bridge._last_stage(
        {"stages": {"photo": {}, "look_photo": {}}}) == "look_photo"
    assert atlas_bridge._last_stage({"stages": {"photo": {}}}) == "photo"

    # "跳过扫描"（triggered=false）没真换过角度：别报成停在观察位扫描，退回 look_photo。
    # 缺 triggered 键的老记录按"扫过"算（{} 那条已经覆盖）。
    assert atlas_bridge._last_stage(
        {"stages": {"photo": {}, "look_photo": {},
                    "observe_scan": {"triggered": False, "notes": ["配置关了"]}}}) == "look_photo"
    assert atlas_bridge._last_stage(
        {"stages": {"photo": {"look_joints_deg": []}, "look_photo": {},
                    "observe_scan": {"triggered": True, "attempts": []}}}) == "observe_scan"

    assert atlas_bridge.Config.from_cfg(
        {"observe_scan": {"enabled": False, "max_views": 2}}).observe_scan == \
        {"enabled": False, "max_views": 2}
    assert atlas_bridge.Config.from_cfg({}).observe_scan == {}
    assert atlas_bridge.Config.from_cfg({"observe_scan": "不是字典"}).observe_scan == {}

    # Pipeline 的接线：Config 里那坨原样传进去（不在桥里合并默认值）
    source = Path(atlas_bridge.__file__).read_text(encoding="utf-8")
    assert "observe_scan_config=config.observe_scan" in source


# ---------------------------------------------------------------------------
# 多锚点：标称位周围扫完还看不准 → 走到清单里的锚点再看、再围它扫
# （锚点来自 observe_scan.anchors_deg；不配就是原来的单锚点行为）
# ---------------------------------------------------------------------------

def _anchor_config(anchor):
    return {"anchors_deg": [list(anchor)]}


def test_multi_anchor_second_anchor_scan_wins(monkeypatch):
    """标称位周围 2 个位姿都看不准 → 走到锚点 1、它自己也不准 → 围它扫，第 2 个看准。"""
    views_a = scan_views(2)
    views_b = [dict(scan_views(1)[0], joints_deg=[50.0, -20.0, 10.0, 5.0, 5.0, 20.0]),
               dict(scan_views(1)[0], joints_deg=[55.0, -25.0, 12.0, 6.0, 6.0, 25.0])]
    stub = StubScan(views_by_call=[views_a, views_b])
    monkeypatch.setattr(pipeline, "observe_scan", stub)
    anchor = [30.0, 60.0, -50.0, 0.0, 20.0, 0.0]
    rig = make_rig(config=_anchor_config(anchor))
    look_script = LookScript(rig, [failure_look(), failure_look(), failure_look(),
                                   failure_look(), failure_look(), ok_look()])
    monkeypatch.setattr(rig.pipeline, "_look", look_script)

    look, stage = rig.pipeline._observe(new_result())

    assert look["ok"] is True
    assert look_script.labels == ["2-look-photo", "2-look-scan-0", "2-look-scan-1",
                                  "2-look-anchor-1", "2-look-anchor1-scan-0",
                                  "2-look-anchor1-scan-1"]
    assert [call["nominal_joints_deg"] for call in stub.calls["plan_views"]] == [
        [float(v) for v in rig.calib.observe_joints], anchor]
    assert rig.client.moves == ([rig.calib.observe_joints]
                                + [view["joints_deg"] for view in views_a]
                                + [anchor]
                                + [view["joints_deg"] for view in views_b])
    assert stage["winning_anchor"] == 1 and stage["winning_index"] == 1
    assert len(stage["anchors"]) == 2
    assert stage["anchors"][0]["look"] is None          # 标称位那一眼在进扫描前就拍过
    assert stage["anchors"][1]["look"]["ok"] is False
    assert stage["anchors"][1]["winning_view"] == 1
    assert [attempt["anchor_index"] for attempt in stage["attempts"]] == [0, 0, 1, 1]
    assert stage["planned"] == views_b                  # 老读法：取"赢的那个锚点"那一份
    assert stage["returned_to_nominal"] is False
    assert stage["triggered"] is True


def test_multi_anchor_anchor_look_succeeds_skips_its_scan(monkeypatch):
    """锚点自己一眼就看准了：不用再围它扫（省 max_views 次走位）。"""
    views_a = scan_views(2)
    stub = StubScan(views=views_a)
    monkeypatch.setattr(pipeline, "observe_scan", stub)
    anchor = [30.0, 60.0, -50.0, 0.0, 20.0, 0.0]
    rig = make_rig(config=_anchor_config(anchor))
    look_script = LookScript(rig, [failure_look(), failure_look(), failure_look(), ok_look()])
    monkeypatch.setattr(rig.pipeline, "_look", look_script)

    look, stage = rig.pipeline._observe(new_result())

    assert look["ok"] is True
    assert look_script.labels == ["2-look-photo", "2-look-scan-0", "2-look-scan-1",
                                  "2-look-anchor-1"]
    assert rig.client.moves == ([rig.calib.observe_joints]
                                + [view["joints_deg"] for view in views_a] + [anchor])
    assert stage["winning_anchor"] == 1
    assert stage["winning_index"] is None
    assert stage["anchors"][1]["planned"] == [] and stage["anchors"][1]["attempts"] == []
    assert len(stub.calls["plan_views"]) == 1          # 没再为锚点周围规划


def test_multi_anchor_all_fail_returns_to_nominal(monkeypatch):
    """标称位 + 一个锚点全看不准：最后回标称位，如实报。"""
    views_a = scan_views(1)
    views_b = [dict(scan_views(1)[0], joints_deg=[50.0, -20.0, 10.0, 5.0, 5.0, 20.0])]
    stub = StubScan(views_by_call=[views_a, views_b])
    monkeypatch.setattr(pipeline, "observe_scan", stub)
    anchor = [30.0, 60.0, -50.0, 0.0, 20.0, 0.0]
    rig = make_rig(config=_anchor_config(anchor))
    look_script = LookScript(rig, [failure_look(), failure_look(), failure_look(),
                                   failure_look()])
    monkeypatch.setattr(rig.pipeline, "_look", look_script)

    look, stage = rig.pipeline._observe(new_result())

    assert look["ok"] is False
    assert rig.client.moves == [rig.calib.observe_joints, views_a[0]["joints_deg"], anchor,
                                views_b[0]["joints_deg"], rig.calib.observe_joints]
    assert stage["winning_anchor"] is None and stage["winning_index"] is None
    assert stage["returned_to_nominal"] is True
    assert len(stage["attempts"]) == 2                 # 标称位周围 1 + 锚点周围 1


def test_multi_anchor_same_as_nominal_is_skipped(monkeypatch):
    """锚点写成跟标称位一样：跳过它（不重复走），按单锚点行为处理。"""
    calib = fakes.FakeCalibration()
    stub = StubScan(views=scan_views(1))
    monkeypatch.setattr(pipeline, "observe_scan", stub)
    rig = make_rig(config={"anchors_deg": [list(calib.observe_joints)]}, calib=calib)
    look_script = LookScript(rig, [failure_look(), failure_look()])
    monkeypatch.setattr(rig.pipeline, "_look", look_script)

    _look, stage = rig.pipeline._observe(new_result())

    assert len(stub.calls["plan_views"]) == 1          # 没有为重复锚点再规划
    assert len(stage["anchors"]) == 1
    assert any("跳过" in note for note in stage["notes"])


def test_multi_anchor_invalid_group_is_skipped_not_fatal(monkeypatch):
    """清单里锚点写坏（不是 6 个数）：跳过它、不炸、照常按标称位处理。"""
    stub = StubScan(views=[])
    monkeypatch.setattr(pipeline, "observe_scan", stub)
    rig = make_rig(config={"anchors_deg": [[1.0, 2.0, 3.0]]})
    monkeypatch.setattr(rig.pipeline, "_look", LookScript(rig, [failure_look()]))

    look, stage = rig.pipeline._observe(new_result())

    assert look["ok"] is False
    assert len(stage["anchors"]) == 1
    assert any("不是 6 个" in note for note in stage["notes"])
    assert rig.client.moves == [rig.calib.observe_joints]


def test_multi_anchor_non_retryable_at_anchor_look_stops(monkeypatch):
    """锚点那一眼遇到不可重试失败（服务坏了）：立刻停，不围它扫、不试后面的锚点。"""
    views_a = scan_views(1)
    stub = StubScan(views_by_call=[views_a, scan_views(1)])
    monkeypatch.setattr(pipeline, "observe_scan", stub)
    rig = make_rig(config={"anchors_deg": [[30.0, 60.0, -50.0, 0.0, 20.0, 0.0],
                                           [40.0, 70.0, -60.0, 10.0, 25.0, 5.0]]})
    look_script = LookScript(rig, [failure_look(), failure_look(),
                                   failure_look(retryable=False, kind="transport",
                                                why="连不上视觉服务")])
    monkeypatch.setattr(rig.pipeline, "_look", look_script)

    look, stage = rig.pipeline._observe(new_result())

    assert look["ok"] is False
    assert look_script.labels == ["2-look-photo", "2-look-scan-0", "2-look-anchor-1"]
    assert stage["winning_anchor"] is None
    assert stage["anchors"][1]["look"]["stopped"] == "失败不可重试，停止扫描"
    assert any("不可重试" in note for note in stage["notes"])
    # 移动过，最后回标称位
    assert rig.client.moves[-1] == [float(v) for v in rig.calib.observe_joints]


def test_multi_anchor_same_as_scan_view_skips_anchor_look(monkeypatch):
    """锚点恰好是标称位周围某个扫描位（前面已经看过）：不再看第二眼，直接围它扫。"""
    views_a = scan_views(2)
    views_b = [dict(scan_views(1)[0], joints_deg=[50.0, -20.0, 10.0, 5.0, 5.0, 20.0])]
    stub = StubScan(views_by_call=[views_a, views_b])
    monkeypatch.setattr(pipeline, "observe_scan", stub)
    anchor = list(views_a[0]["joints_deg"])            # 就是标称位扫描的第 1 个位姿
    rig = make_rig(config={"anchors_deg": [anchor]})
    look_script = LookScript(rig, [failure_look(), failure_look(), failure_look(),
                                   failure_look()])
    monkeypatch.setattr(rig.pipeline, "_look", look_script)

    look, stage = rig.pipeline._observe(new_result())

    assert look["ok"] is False
    # 没有 "2-look-anchor-1" 这一眼：锚点那个位姿前面已经看过了
    assert look_script.labels == ["2-look-photo", "2-look-scan-0", "2-look-scan-1",
                                  "2-look-anchor1-scan-0"]
    assert stage["anchors"][1]["look"] == {"skipped": "这个位姿前面已经看过，不再看第二眼"}
    assert any("跳过它自己那一照" in note for note in stage["notes"])
    assert rig.client.moves == ([rig.calib.observe_joints]
                                + [view["joints_deg"] for view in views_a]
                                + [views_b[0]["joints_deg"]]
                                + [rig.calib.observe_joints])


def test_multi_anchor_reestimates_surface_from_its_own_frame(monkeypatch):
    """每个锚点用它自己那一帧重估支撑面。

    真机实测：标称位那帧看到的是地面（z≈-0.47 m），人记录的第二个观察位那帧看到的是
    桌面（z≈+0.28 m）。围锚点扫的位姿必须按桌面高度规划，否则全打到地面上。
    """
    views_a = scan_views(1)
    views_b = [dict(scan_views(1)[0], joints_deg=[50.0, -20.0, 10.0, 5.0, 5.0, 20.0])]
    stub = StubScan(views_by_call=[views_a, views_b],
                    surface_z_sequence=[-0.457, 0.279])
    monkeypatch.setattr(pipeline, "observe_scan", stub)
    anchor = [30.0, 60.0, -50.0, 0.0, 20.0, 0.0]
    rig = make_rig(config={"anchors_deg": [anchor]})
    look_script = LookScript(rig, [failure_look(), failure_look(), failure_look(), ok_look()])
    monkeypatch.setattr(rig.pipeline, "_look", look_script)

    look, stage = rig.pipeline._observe(new_result())

    assert look["ok"] is True
    assert [call["surface_z"] for call in stub.calls["plan_views"]] == \
        [pytest.approx(-0.457), pytest.approx(0.279)]
    assert stage["anchors"][0]["surface_z_m"] == pytest.approx(-0.457)
    assert stage["anchors"][1]["surface_z_m"] == pytest.approx(0.279)
    assert any("重估支撑面高度" in note for note in stage["notes"])


def test_multi_anchor_surface_falls_back_to_nominal_frame(monkeypatch):
    """锚点那一帧估不出支撑面：沿用标称位那帧的值（并写明）。"""
    views_a = scan_views(1)
    views_b = [dict(scan_views(1)[0], joints_deg=[50.0, -20.0, 10.0, 5.0, 5.0, 20.0])]
    stub = StubScan(views_by_call=[views_a, views_b],
                    surface_z_sequence=[-0.457, None])
    monkeypatch.setattr(pipeline, "observe_scan", stub)
    anchor = [30.0, 60.0, -50.0, 0.0, 20.0, 0.0]
    rig = make_rig(config={"anchors_deg": [anchor]})
    look_script = LookScript(rig, [failure_look(), failure_look(), failure_look(), ok_look()])
    monkeypatch.setattr(rig.pipeline, "_look", look_script)

    _look, stage = rig.pipeline._observe(new_result())

    assert [call["surface_z"] for call in stub.calls["plan_views"]] == \
        [pytest.approx(-0.457), pytest.approx(-0.457)]
    assert stage["anchors"][1]["surface_z_m"] == pytest.approx(-0.457)
    assert any("沿用上一个面高" in note for note in stage["notes"])
