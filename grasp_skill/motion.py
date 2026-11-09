# Description: 把计划变成动作：挑一个腕关节够得着的闭合方向，解 IK，然后走下去。
#
# 这里只用到 calib 对象的两个方法（solve、tip 之类），所以真机和仿真共用同一份代码——
# 真机那份在 calib_io.Calibration，仿真那份在 sim/kinematics.SimCalibration。
#
# 为什么要有"挑闭合方向"这一步：物体长轴定出的首选方向，腕关节不一定转得到。绕接近轴换
# 闭合方向等价于换手腕转角，J5/J6 的限位直接卡在这上面。实测同一个目标、同样竖直朝下，
# 有的转角 0 mm 到位，有的差 20 mm 以上。所以按"离首选多近"从近到远逐个试，够得着就用，
# 并把实际偏离记下来（偏得多了就是斜着夹，抓不稳）。
import numpy as np

import grasp

# 挑方向时的步长与最大偏离
CLOSING_STEP_DEG = 30.0
# 闭合方向**不搜索**——只对准短轴（等价于"垂直于物体长轴"）。理由是第一性的：
# 夹爪绕工具轴是 360° 的，转腕子只改手指朝哪边合，**不改爪尖在哪**（前提：爪尖在工具轴上，
# 见标定那一步——以前它偏轴 10 mm，转腕子会挪爪尖，才把"方向"和"够不够得着"搅在一起，
# 于是这里长出一套"绕着短轴左右试方向"的搜索）。方向是方向，够不够得着是位置的事，
# 分开之后：悬停和下爪都对准短轴 → 姿态一致 → 下爪路上不需要转腕子 →
# 也不会再被"物体上方不许大幅转动"那条闸拦住。
# 真要站不到那个腕角（关节限位卡住），就该如实报"够不着"，而不是悄悄歪着夹。
CLOSING_LIMIT_DEG = 0.0
TILT_STEPS_DEG = (0.0, 5.0, 10.0, 20.0, 30.0, 45.0)

# 爪尖到位判据：实际停的地方离目标不能超过这么多（三维直线距离，米）。
# 比计划侧允许的横向容差（grasp.LATERAL_TOL_M，5 mm）严，两个量测的不是一回事：
# 这个量"有没有走到命令的位置"，那个量"这个位置还能不能夹住"。
MOVE_TOLERANCE_M = 0.0015


def plan_reach(calib, plan, tilt_limit_deg: float = 0.0,
               step_deg: float = CLOSING_STEP_DEG, n_starts: int = 12,
               validate=None, closing_tol_deg=None,
               lateral_tol_m=None, along_tol_m=None) -> dict:
    """给计划找一个腕关节到得了的位姿。

    先按首选姿态（严格垂直于平面）试各个闭合方向，够不着就往"朝着基座放斜"的方向逐档放，
    每档再试一遍闭合方向。放斜多少算够得着由调用方定：

    - 抓取那一刻：`tilt_limit_deg=0`，姿态偏差只能落在 grasp 那套限值里（默认 12°）。
    - 悬停/路过：可以放宽，悬停只是路过，倾斜不影响抓取质量。

    validate: 可选的加试，`validate(solved) -> bool`。站得住还不够时用——悬停位除了
        自己够得着，还得能从那儿下爪（两支臂型不同的话，下爪要整条臂重排 100°）。
        返回假就继续试下一个闭合方向。

    Returns:
        {"ok", "closing", "closing_offset_deg", "approach", "tilt_deg", "solved",
         "verdict", "tried"}；够不着时 `why` 写原因。
    """
    if not plan["ok"]:
        return {"ok": False, "why": "计划本身没通过：" + "；".join(plan["problems"]),
                "closing": plan["closing"], "closing_offset_deg": 0.0, "tilt_deg": 0.0,
                "approach": plan["approach"], "solved": None, "verdict": None, "tried": []}

    target = np.asarray(plan["grasp_tip"], dtype=float)
    tried = []
    best = None
    validate_rejected = 0        # 站得住、但加试没过的候选数（调用方好据此说清原因）
    for tilt in [t for t in TILT_STEPS_DEG if t <= tilt_limit_deg] or [0.0]:
        # 放斜有两个方向：朝基座仰、朝外倾。哪个方向有用取决于目标远离还是靠近底座——
        # 朝基座仰等于把腕往外送，目标本来就在近处时反而更够不着。两个都试。
        signs = (0,) if tilt == 0 else (1, -1)
        for sign in signs:
            approach = _tilted_approach(plan, tilt, sign)
            for closing, offset in grasp.candidate_closings({**plan, "approach": approach.tolist()},
                                                            step_deg, CLOSING_LIMIT_DEG):
                candidate = grasp.rotation_from_axes(approach, closing)
                solved = calib.solve(target, candidate, n_starts=n_starts)
                if not solved.get("ok", False):
                    tried.append({"tilt_deg": tilt, "tilt_sign": sign, "offset_deg": offset,
                                  "why": solved.get("message", "")})
                    continue
                verdict = grasp.evaluate_reach(
                    plan, solved, max_obliquity_deg=tilt_limit_deg or grasp.MAX_OBLIQUITY_DEG,
                    **({} if closing_tol_deg is None else {"closing_tol_deg": closing_tol_deg}),
                    lateral_tol_m=lateral_tol_m, along_tol_m=along_tol_m)
                tried.append({"tilt_deg": tilt, "tilt_sign": sign, "offset_deg": offset,
                              "position_error_m": solved["position_error_m"],
                              "obliquity_deg": verdict["obliquity_deg"],
                              "lateral_miss_m": verdict["lateral_miss_m"]})
                score = (verdict["lateral_miss_m"] / grasp.LATERAL_TOL_M
                         + abs(verdict["along_miss_m"]) / grasp.ALONG_TOL_M
                         + tilt / 45.0)          # 方向不再参与打分：只有一个候选（短轴）
                if best is None or score < best[0]:
                    best = (score, closing, offset, approach, tilt, sign, solved, verdict)
                if verdict["ok"]:
                    if validate is None or validate(solved):
                        # 只要有一个候选过关就停：候选是按"离首选多近"排好序的，先过的就是最贴的。
                        # 不这样，一个够得着的位姿也要把这一档的 12 个闭合方向全试完（实测 173 s）。
                        break
                    validate_rejected += 1
        if best is not None and best[7]["ok"] and best[4] == tilt:
            break                       # 这一档已经够得着了，不再往更斜的档试
    if best is None:
        return {"ok": False, "why": "各个方向都解不出 IK。", "closing": plan["closing"],
                "closing_offset_deg": 0.0, "tilt_deg": 0.0, "approach": plan["approach"],
                "solved": None, "verdict": None, "tried": tried}
    _, closing, offset, approach, tilt, sign, solved, verdict = best
    result = {"ok": verdict["ok"], "closing": closing.tolist(), "closing_offset_deg": offset,
              "approach": approach.tolist(), "tilt_deg": tilt, "tilt_sign": sign,
              "solved": solved, "verdict": verdict, "tried": tried,
              "validate_rejected": validate_rejected}
    if not verdict["ok"]:
        result["why"] = "；".join(verdict["reasons"])
    else:
        notes = []
        if offset:
            notes.append("闭合方向绕接近轴偏了 %.0f°（首选方向腕关节转不到）" % offset)
        if tilt:
            notes.append("工具轴%s放斜 %.0f°（严格垂直于平面够不着，这一档才到位）"
                         % ("朝基座" if sign > 0 else "朝外", tilt))
        if notes:
            result["note"] = "；".join(notes) + "。"
    return result


def move_tip(calib, move, target_tip, rotation, rounds: int = 3,
             tolerance_m: float = MOVE_TOLERANCE_M, n_starts: int = 12,
             max_joint_move_deg: float | None = None,
             settled_m: float | None = None) -> dict:
    """把爪尖送到指定位置：解 IK → 走过去 → 量实际停在哪儿 → 差了再补一刀。

    为什么要补：位置伺服有稳态误差。仿真实测命令 J5=49.13°，实际停在 50.84°，
    1.7° 在 0.3 m 外就是 11 mm——算法算得再准，那 11 mm 也照样落不到物体上。
    于是走完读一次关节角（臂自己报的那份），把残差加到下一次的目标上再走一遍。

    Args:
        calib: 有 solve / flange / tip 的标定对象（真机、仿真同一接口）。
        move: 回调 `move(joints_deg) -> 实际关节角（度）`；真机是 arm.move_joints + arm.joints()，
            仿真是设执行器 + 等它稳住 + 读回来。
        target_tip: 爪尖目标位置（基座系，米）。
        rotation: 目标姿态 3x3。
        max_joint_move_deg: 只认"关节没走超过这个角度"的解，透传给 calib.solve；
            目前只有仿真那份（SimCalibration.solve）支持，真机那份传了会报错。

    Returns:
        {"ok", "rounds", "error_m", "history", "solved", "joints_deg"}。
    """
    target = np.asarray(target_tip, dtype=float)
    command = target.copy()
    history = []
    solved = None
    solve_kwargs = {}
    if max_joint_move_deg is not None:
        solve_kwargs["max_joint_move_deg"] = max_joint_move_deg
    if settled_m is not None:
        solve_kwargs["settled_m"] = settled_m
    for round_index in range(max(1, rounds)):
        solved = calib.solve(command, rotation, n_starts=n_starts, **solve_kwargs)
        solve_debug = getattr(calib, "last_solve_debug", None)
        if not solved.get("ok", False):
            return {"ok": False, "rounds": round_index, "error_m": None, "history": history,
                    "solved": solved, "why": solved.get("message", "IK 不收敛")}
        actual_joints = move(solved["joints_deg"])
        achieved = calib.tip(calib.flange(actual_joints))
        error = target - achieved
        distance = float(np.linalg.norm(error))
        history.append({"round": round_index, "error_m": distance,
                        "target_tip_m": target.tolist(), "command_tip_m": command.tolist(),
                        "achieved_tip_m": np.asarray(achieved, dtype=float).tolist(),
                        "commanded_joints_deg": list(np.asarray(solved["joints_deg"], dtype=float)),
                        "joints_deg": list(np.asarray(actual_joints, dtype=float)),
                        "ik_position_error_m": solved.get("position_error_m"),
                        "ik_tilt_error_deg": solved.get("tilt_error_deg"),
                        "ik_joint_move_deg": solved.get("joint_move_deg"),
                        "ik_debug": solve_debug})
        if distance <= tolerance_m:
            break
        command = command + error          # 把残差补到下一次的目标上
    return {"ok": distance <= tolerance_m, "rounds": len(history), "error_m": distance,
            "history": history, "solved": solved, "joints_deg": history[-1]["joints_deg"]}


def _tilted_approach(plan, tilt_deg: float, sign: int = 1) -> np.ndarray:
    """把接近方向绕平面内偏 tilt_deg：sign=+1 朝基座那一侧仰，-1 朝外倾。"""
    approach = np.asarray(plan["approach"], dtype=float)
    approach = approach / np.linalg.norm(approach)
    if tilt_deg <= 0:
        return approach
    plane_point = np.asarray(plan["plane"]["point"], dtype=float)
    target = np.asarray(plan["grasp_tip"], dtype=float)
    normal = np.asarray(plan["plane"]["normal"], dtype=float)
    normal = normal / np.linalg.norm(normal)
    lean = -target - plane_point * 0.0          # 从目标指向基座原点（基座原点在原点）
    lean = lean - float(lean @ normal) * normal
    if float(np.linalg.norm(lean)) < 1e-9:
        return approach
    lean = lean / np.linalg.norm(lean)
    if sign < 0:
        lean = -lean
    axis = np.cross(approach, lean)
    if float(np.linalg.norm(axis)) < 1e-9:
        return approach
    axis = axis / np.linalg.norm(axis)
    angle = np.deg2rad(tilt_deg)
    rotated = (approach * np.cos(angle)
               + np.cross(axis, approach) * np.sin(angle)
               + axis * float(axis @ approach) * (1 - np.cos(angle)))
    return rotated / np.linalg.norm(rotated)
