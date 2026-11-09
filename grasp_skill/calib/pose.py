# Description: 把臂报的末端位姿拼成 4x4——手眼和夹爪尖两页都要用。
#
# 单独放一处是因为欧拉角的约定容易搞错：臂报的是 [RZ, RY, RX]（见
# common/robot/direct.py 的 end_euler_zyx_deg），顺序反了旋转矩阵就错了，
# 而错了之后标出来的结果照样有个数，只是不对。
import json

import numpy as np
from scipy.spatial.transform import Rotation

from common import handeye3d
from common.robot import kinematics


def flange_transform(pos, euler_zyx_deg) -> np.ndarray:
    """基座系下的法兰位姿。

    Args:
        pos: [x, y, z]，米（DirectArm.end_pos 的返回值）。
        euler_zyx_deg: [RZ, RY, RX]，度（DirectArm.end_euler_zyx_deg 的返回值）。
    """
    rotation = Rotation.from_euler("zyx", euler_zyx_deg, degrees=True).as_matrix()
    return handeye3d.make_transform(rotation, pos)


_FORWARD_CACHE: dict = {}


def corrected_forward(correction):
    """按修正取一个 fk(joints_deg) -> 4x4；没有修正（或修正恒等）返回 None。

    建链要解析一遍 URDF，几十毫秒，所以按修正内容缓存一份。
    """
    if kinematics.is_identity(correction):
        return None
    key = json.dumps([correction["zeros_deg"], correction["offsets_m"]], sort_keys=True)
    if key not in _FORWARD_CACHE:
        from common.robot.ik import build_chain_and_bounds

        chain, _ = build_chain_and_bounds(correction=correction)
        _FORWARD_CACHE.clear()
        _FORWARD_CACHE[key] = chain.forward_kinematics
    return _FORWARD_CACHE[key]


def flange_of(joints_deg, pos, euler_zyx_deg, correction=None) -> np.ndarray:
    """法兰位姿：有几何修正就走修正后的正运动学，没有就用臂上报的位姿。

    两者本来只差 0.08 mm（实测），差的正是修正要补的那部分，所以换过去不会引入别的东西。
    修正是从多机位手眼数据里拟合出来的，见 common/robot/kinematics.py。
    """
    forward = corrected_forward(correction)
    if forward is None or not joints_deg:
        return flange_transform(pos, euler_zyx_deg)
    return np.asarray(forward(np.deg2rad(kinematics.corrected_joints(joints_deg, correction))).matrix())
