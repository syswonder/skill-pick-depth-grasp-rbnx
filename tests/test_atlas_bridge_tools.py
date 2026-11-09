# Description: atlas_bridge 三个工具的**编排**测试（put_down 顺序 / pick·observe 响应组装）。
#
# 不连 ROS / atlas / 视觉服务：
#   - `_put_down` 只碰 `_state.client`（关节命令与夹爪）和 `_state.pipeline.opening_m()`，
#     这里用两个假对象替换掉，断言"先到放点 → 开到最大 → 回安全位"的顺序；
#   - `pick` / `observe` 只负责把 `_run` 的结果翻译成契约响应，替换 `_run` 就能黑盒测。
#
# 这个文件需要 codegen 产物 + robonix_api 才能 import `grasp_skill.atlas_bridge`；
# 裸跑 `pytest tests/`（没给 PYTHONPATH）时整个文件 skip，不算失败。
import json

import pytest

pytest.importorskip("robonix_api")
pytest.importorskip("pick_mcp")

from grasp_skill import atlas_bridge                    # noqa: E402
from pick_mcp import Observe_Request, Pick_Request      # noqa: E402


class FakeClient:
    def __init__(self):
        self.calls = []
        self.joints = [0.0] * 6
        self.gripper_m = 0.102

    def move_joints(self, joints_deg, **kwargs):
        del kwargs
        self.calls.append(("move", [float(v) for v in joints_deg]))
        self.joints = [float(v) for v in joints_deg]
        return list(self.joints)

    def gripper(self, opening_m, **kwargs):
        del kwargs
        self.calls.append(("gripper", float(opening_m)))
        self.gripper_m = float(opening_m)
        return self.gripper_m


class FakePipeline:
    def opening_m(self):
        return 0.056

    def run(self):
        raise AssertionError("编排测试不该真的跑抓取")


@pytest.fixture
def state(monkeypatch):
    client = FakeClient()
    pipeline = FakePipeline()
    monkeypatch.setattr(atlas_bridge._state, "client", client)
    monkeypatch.setattr(atlas_bridge._state, "pipeline", pipeline)
    return client, pipeline


def test_put_down_goes_to_drop_point_opens_gripper_then_safe_pose(state):
    client, _pipeline = state
    config = atlas_bridge.Config(put_down_joints_deg=[10.0, 20, 30, 40, 50, 60],
                                 teach_safe_joints_deg=[-1, -2, -3, -4, -5, -6])
    ok, message = atlas_bridge._put_down(config)
    assert ok is True
    assert client.calls == [
        ("move", [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]),
        ("gripper", pytest.approx(0.056)),
        ("move", [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0]),
    ]
    assert "放下完成" in message


def test_put_down_without_drop_point_releases_in_place_then_home(state):
    client, _pipeline = state
    ok, message = atlas_bridge._put_down(atlas_bridge.Config())
    assert ok is True
    # 没配 put_down_joints_deg：不动臂，直接开爪；然后回全零 home。
    assert client.calls == [
        ("gripper", pytest.approx(0.056)),
        ("move", [0.0] * 6),
    ]
    assert "就在当前位放开" in message


def test_put_down_inactive_reports_failure(monkeypatch):
    monkeypatch.setattr(atlas_bridge._state, "client", None)
    monkeypatch.setattr(atlas_bridge._state, "pipeline", None)
    ok, message = atlas_bridge._put_down(atlas_bridge.Config())
    assert ok is False
    assert "还没激活" in message


def test_pick_response_assembles_contract_fields(monkeypatch):
    result = {
        "ok": True,
        "problems": [],
        "stages": {
            "descend": {"joints_deg": None},           # 没走到下爪 → 零位姿
            "close": {"final_opening_mm": 42.6},
        },
    }
    monkeypatch.setattr(atlas_bridge, "_run",
                        lambda action, target="": (True, "lift", "抓住了", result))

    response = atlas_bridge.pick(Pick_Request(object_name="红色方块"))

    assert response.success is True
    assert response.message == "抓住了"
    assert response.gripper_width == pytest.approx(0.0426)
    assert response.score == 0.0
    assert response.elapsed_s >= 0.0
    pose = response.grasp_pose.to_dict()
    assert pose["header"]["frame_id"] == "arm/base_link"
    assert pose["pose"]["position"] == {"x": 0.0, "y": 0.0, "z": 0.0}
    assert pose["pose"]["orientation"]["w"] == 1.0


def test_pick_failure_message_starts_with_stage(monkeypatch):
    monkeypatch.setattr(
        atlas_bridge, "_run",
        lambda action, target="": (False, "descend", "爪尖没下到计划位置", {}))
    response = atlas_bridge.pick(Pick_Request(object_name="cube"))
    assert response.success is False
    assert response.message == "descend: 爪尖没下到计划位置"
    assert response.grasp_pose.to_dict()["pose"]["position"] == {"x": 0.0, "y": 0.0, "z": 0.0}


def test_observe_response_carries_detail_json(monkeypatch):
    result = {"ok": True, "action": "observe", "problems": [],
              "stages": {"photo": {"joints_deg": [1, 2, 3, 4, 5, 6]}},
              "measurement": {"top_point_m": [0.1, 0.2, 0.3]}}
    monkeypatch.setattr(atlas_bridge, "_run",
                        lambda action, target="": (True, "observe", "看完一眼", result))

    response = atlas_bridge.observe(Observe_Request(target="cube"))

    assert response.success is True
    assert response.message == "看完一眼"
    assert json.loads(response.detail_json)["measurement"]["top_point_m"] == [0.1, 0.2, 0.3]
