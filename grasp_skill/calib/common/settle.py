# Description: 等臂停稳再拍——动完立刻拍，抖动会同时污染角点和臂上报的位姿。
#
# 手眼那条方程 A·X = X·B 里，A 是臂上报的法兰位姿、B 是角点解出的板子位姿。抖动期间
# 两者都在变，而且不同步：相机曝光的那一瞬间跟 CAN 回读的时刻对不上。所以抖动不是把
# 误差放大，是让 A 和 B 对不上号，解出来的 X 直接偏掉，而残差看不出这件事。
#
# 判据：连着 STILL_WINDOW_S 秒里，关节角的峰峰值够小，同时同一个角点在画面上的位置够稳。
# 两个都要过，因为它们各自漏一头——臂报到位了相机还可能在晃，角点清楚了也不代表
# 关节读数已经稳定。
#
# 窗口不能只看相邻两帧差多少。这条臂停下来是衰减振荡，相邻两帧可能正好落在波的顶点附近，
# 一帧差不了几个像素，看着像停了。要看整段窗口里的位移，振荡才露得出来。
#
# 角点用"相对自身均值的均方根"而不是峰峰值：峰峰值会随窗口变长而变大，窗口从 3 帧拉到
# 几十帧以后，光静态检测噪声就能把一个固定阈值顶穿，判据会变成永远过不了。
import time

import numpy as np

from common.calib_log import get_logger

log = get_logger("settle")

STILL_WINDOW_S = 1.0        # 要连着这么久都稳才算停稳
STILL_MIN_FRAMES = 12       # 窗口里至少凑够这么多帧（相机慢时帧数不够就再等）
JOINT_STILL_DEG = 0.05      # 窗口里关节角的峰峰值上限
CORNER_STILL_PX = 1.0       # 窗口里角点相对自身均值的均方根上限（静态检测噪声约 0.3px）
SETTLE_TIMEOUT_S = 20.0     # 这条臂停下来要抖好几秒，等够了再说停不下来


def settle_detect(detect, color):
    """包装 detect：出错或返回空都不让 wait_until_still 卡死。"""
    try:
        corners, ids = detect(color)
        if corners is None or ids is None or len(ids) == 0:
            return None, None
        return corners, ids
    except Exception:
        return None, None


def _as_pairs(corners, ids):
    """把检测结果整成 (ids_flat, Nx2)；没检出就返回 None。"""
    if corners is None or ids is None or len(ids) == 0:
        return None
    return np.asarray(ids).ravel(), np.asarray(corners).reshape(-1, 2)


def _corner_steadiness_px(window: list) -> float:
    """窗口里角点位置的稳定程度：每个编号相对自身均值的均方根偏移，取中位数。

    按编号配对而不是按顺序，因为每帧认出的角点个数会变，按顺序会把不同的角点比到一起。
    只统计在窗口里出现过七成以上的编号，偶尔冒出来的那几个本身就不稳，不该拖累判据。

    Returns:
        像素；窗口里没跟住任何角点时返回 nan，此时调用方只能认关节那一头。
    """
    track: dict[int, list] = {}
    for pairs in window:
        if pairs is None:
            continue
        for index, point in zip(pairs[0], pairs[1]):
            track.setdefault(int(index), []).append(point)
    need = max(2, int(len(window) * 0.7))
    deviations = []
    for points in track.values():
        if len(points) < need:
            continue
        array = np.asarray(points, dtype=float)
        deviations.append(float(np.sqrt(((array - array.mean(axis=0)) ** 2).sum(axis=1).mean())))
    return float(np.median(deviations)) if deviations else float("nan")


def wait_until_still(camera, arm, detect=None, timeout_s: float = SETTLE_TIMEOUT_S) -> dict:
    """一直取帧，直到臂和画面都静够 STILL_WINDOW_S 秒，把静下来后那一帧连检测结果一起返回。

    帧是一帧一帧现取的（RealSense 的 wait_for_frames 会等下一帧），所以相邻两帧真的相邻。

    Args:
        camera: 有 get_frames() 的相机。
        arm: 有 joints() 的臂。
        detect: 可选回调 detect(color) -> (corners, ids)；给了就一并查角点稳不稳，
            并把结果带回来（省得调用方再检一遍）。
        timeout_s: 等这么久还不稳就返回 ok=False。

    Returns:
        {"ok", "frame", "color", "corners", "ids", "waited_s", "joint_ptp_deg",
        "corner_rms_px", "why"}；frame 是整个取帧结果（深度反投影要用里面的深度帧）。
        不 ok 时 frame/color/corners/ids 全是 None：抖动期间的帧一张都不能用，
        返回了迟早有人拿去解算。
    """
    start = time.time()
    collect_start = None       # 第一帧落袋的时刻，"看够一整段"从这儿算
    window: list[tuple[float, np.ndarray, tuple | None]] = []
    corners = ids = None
    saw_color = False
    frame_index = 0            # 检测隔帧跑：每帧都检会拖慢取帧，稳定窗口反而凑不满
    calmest_joint, calmest_corner = float("inf"), float("inf")   # 最稳到过什么程度，超时时报出来调阈值
    while time.time() - start < timeout_s:
        frame = camera.get_frames()
        color = frame.get("color")
        if color is None:
            continue
        saw_color = True
        # 关节读数紧跟取帧，别让检测的耗时把两者拉开——抖动期间这两个时刻的间隔就是误差
        joints = np.asarray(arm.joints(), dtype=float)
        pairs = None
        if detect is not None:
            frame_index += 1
            if frame_index % 2 == 0:
                corners, ids = settle_detect(detect, color)
            pairs = _as_pairs(corners, ids) if corners is not None else None
        now = time.time()
        if collect_start is None:
            collect_start = now
        window.append((now, joints, pairs))
        window = [item for item in window if now - item[0] <= STILL_WINDOW_S]
        if now - collect_start < STILL_WINDOW_S or len(window) < STILL_MIN_FRAMES:
            continue          # 还没看够一整段，这时候的"稳"只是看得不够久
        joint_ptp = float(np.ptp(np.stack([item[1] for item in window]), axis=0).max())
        corner_rms = _corner_steadiness_px([item[2] for item in window])
        calmest_joint = min(calmest_joint, joint_ptp)
        if not np.isnan(corner_rms):
            calmest_corner = min(calmest_corner, corner_rms)
        if joint_ptp <= JOINT_STILL_DEG and (np.isnan(corner_rms) or corner_rms <= CORNER_STILL_PX):
            log.debug("停稳: waited=%.1fs joint_ptp=%.3f° corner_rms=%.2fpx",
                      time.time() - start, joint_ptp, corner_rms)
            return {"ok": True, "frame": frame, "color": color, "corners": corners, "ids": ids,
                    "waited_s": time.time() - start, "joint_ptp_deg": joint_ptp,
                    "corner_rms_px": corner_rms, "why": ""}
    # 超时报的是等待期间最稳到过什么程度，不是最后一帧——臂在振荡的话最后一帧可能正赶上
    # 波峰，看着像完全没停下来，就判断不出阈值是不是定得太紧。
    waited = time.time() - start
    log.info("wait_until_still 超时: 等了 %.1fs，saw_color=%s calmest_joint=%.3f° calmest_corner=%.2fpx -> %s",
             waited, saw_color,
             calmest_joint if calmest_joint < float("inf") else float("nan"),
             calmest_corner if calmest_corner < float("inf") else float("nan"),
             "一直取不到画面" if not saw_color else
             ("%.0f 秒里凑不够连续 %d 帧" % (timeout_s, STILL_MIN_FRAMES)
              if calmest_joint == float("inf") else "还在抖（关节 %.3f° 角点 %.2fpx）"
              % (calmest_joint, calmest_corner)))
    if not saw_color:
        why = "一直取不到画面"
    elif calmest_joint == float("inf"):
        why = "%.0f 秒里凑不够连续 %d 帧，相机太慢" % (timeout_s, STILL_MIN_FRAMES)
    else:
        why = "等了 %.1f 秒还在抖：关节峰峰值最稳时也有 %.3f°（要 %.2f 以内）" % (
            waited, calmest_joint, JOINT_STILL_DEG)
        if calmest_corner < float("inf"):
            why += "，角点抖最稳时 %.2f px（要 %.2f 以内）" % (calmest_corner, CORNER_STILL_PX)
    nan = float("nan")
    return {"ok": False, "frame": None, "color": None, "corners": None, "ids": None, "waited_s": waited,
            "joint_ptp_deg": calmest_joint if calmest_joint < float("inf") else nan,
            "corner_rms_px": calmest_corner if calmest_corner < float("inf") else nan, "why": why}
