# Description: 抓不到（够不着）时的退避重试：回 home → 站立 → 前进一小段 → 趴下 → 再抓。
#
# 为什么要单独一层：`pipeline.py` 只认臂和腕上相机，不知道底盘；而底盘动作要经
# atlas 解析出来的 MCP 端点（`grasp_skill/chassis.py`）。这里只依赖鸭子类型：
#   pipeline.run() -> dict             一轮抓取
#   client.move_joints([...])          臂回 home
#   chassis.set_posture("stand"/"sit") 底盘姿态（返回 (ok, message)）
#   chassis.move_forward(m)            前进（返回 (ok, message)）
# 所以 tests/test_backoff.py 可以完全不碰 ROS / atlas。
#
# 只对"够不着"重试（pipeline 在结果里标 `failure_kind == "out_of_reach"`）：
# 看不清、量不准、被挡住这些情况，把狗往前挪 10 cm 只会把事情弄得更糟。
from __future__ import annotations

import logging
import time

log = logging.getLogger("depth_grasp.backoff")

REACH_FAILURE = "out_of_reach"
HOME_JOINTS_DEG = [0.0] * 6
# 回 home 之后才能站起来：差这么多度以上就认为没到位（示教模式、被挡住、CAN 掉了）。
HOME_TOL_DEG = 5.0


def run_with_backoff(pipeline, client, chassis, *,
                     max_attempts: int = 1,
                     forward_m: float = 0.10,
                     settle_s: float = 0.5,
                     home_on_failure: bool = True) -> dict:
    """跑一轮抓取；够不着就退避后重试，最多 `max_attempts` 次。

    每次退避的顺序（用户要求）：臂先回 home → 站立（底盘移动要求 RL 控制）→ 前进
    `forward_m` → 趴下（回到抓取标定时的姿态）→ 重跑 observe/plan/grasp。

    返回值就是最后一轮 `pipeline.run()` 的结果，附加 `stages["backoff"]`：
        {"attempts": [{"attempt": 1, "forward_m": 0.10, "arm_home": "ok",
                       "stand": "...", "move": "done", "sit": "...",
                       "result": "ok" | "out_of_reach" | "aborted", ...}]}
    """
    attempts: list[dict] = []
    result = pipeline.run()
    attempt = 0
    while (not result.get("ok")
           and result.get("failure_kind") == REACH_FAILURE
           and attempt < max_attempts):
        attempt += 1
        record: dict = {"attempt": attempt, "forward_m": float(forward_m),
                        "before_problems": list(result.get("problems") or [])}
        attempts.append(record)
        log.info("够不着：第 %d/%d 次退避（回 home → 前进 %.2f m → 再抓）",
                 attempt, max_attempts, forward_m)
        ok, message = _nudge(client, chassis, forward_m, settle_s, record)
        if not ok:
            record["result"] = "aborted"
            record["message"] = message
            log.warning("退避中止：%s", message)
            break
        result = pipeline.run()
        record["result"] = ("ok" if result.get("ok")
                            else (result.get("failure_kind") or "failed"))
        record["problems"] = list(result.get("problems") or [])
        log.info("退避后第 %d 轮结果：%s", attempt, record["result"])

    if attempts:
        result.setdefault("stages", {})["backoff"] = {"attempts": attempts}
    if (home_on_failure and not result.get("ok")
            and result.get("failure_kind") == REACH_FAILURE):
        # 不管还有没有下一次，失败后都把臂收回来：底盘/导航接着动的时候，
        # 臂举在拍照位或悬停位是不安全的。
        _home(client, result)
    return result


def _nudge(client, chassis, forward_m: float, settle_s: float,
           record: dict) -> tuple[bool, str]:
    """回 home → stand → move → sit。每步都写进 record，失败就停。"""
    ok, message = _move_home(client)
    record["arm_home"] = message
    if not ok:
        return False, message

    if chassis is None:
        return False, "底盘能力没解析到（atlas 上没有 chassis/move + chassis/posture）"

    ok, message = chassis.set_posture("stand")
    record["stand"] = message
    if not ok:
        return False, f"站立失败：{message}"

    ok, message = chassis.move_forward(forward_m)
    record["move"] = message
    if not ok:
        # 别把机器人留在站着的状态：尽量趴回去，再报错。
        try:
            chassis.set_posture("sit")
        except Exception:                          # noqa: BLE001
            pass
        return False, f"前进 {forward_m:.2f} m 失败：{message}"

    time.sleep(max(0.0, settle_s))
    ok, message = chassis.set_posture("sit")
    record["sit"] = message
    if not ok:
        return False, f"趴下失败：{message}"
    time.sleep(max(0.0, settle_s))
    return True, "ok"


def _move_home(client) -> tuple[bool, str]:
    """Send home and verify the arm actually got there."""
    try:
        actual = client.move_joints(list(HOME_JOINTS_DEG))
    except Exception as error:                     # noqa: BLE001 —— 由调用方如实上报
        return False, f"回 home 失败：{type(error).__name__}: {error}"
    if actual is None:
        return False, "回 home 失败：没有关节反馈"
    error = max(abs(float(a) - b) for a, b in zip(actual, HOME_JOINTS_DEG))
    if error > HOME_TOL_DEG:
        return False, (f"回 home 没到位（最大差 {error:.1f}°，判据 "
                       f"{HOME_TOL_DEG:.0f}°）——示教模式、被挡住或 CAN 异常？")
    return True, "ok"


def _home(client, result: dict) -> None:
    """Best-effort return to home; failures go into the result, not an exception."""
    ok, message = _move_home(client)
    result.setdefault("stages", {})["backoff_home"] = message
    if not ok:
        result.setdefault("problems", []).append("退避后" + message)
