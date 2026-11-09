# Description: 臂的几何修正——关节零点与各关节的小平移，连同"用法兰位姿的那几处"一起收在这。
#
# 为什么需要它：URDF 是厂家给的标称模型，实物总有出入。实测这条 piper：拿它当模型，
# 同一块没动过的板子在不同位姿下算出来的位置差 22 mm；放开手腕三个关节的零点之后降到
# 2.8 mm，而且这组数在完全没参与拟合的位姿上照样管用——说明误差确实在模型里，不在噪声里。
#
# 修正的定义（都在关节的父坐标系里，且都是小量）：
#   零点 zeros_deg[i]  —— 第 i 个关节的真实角度 = 编码器读数 + 这个值（度）。
#   平移 offsets_m[i]  —— 第 i 个关节之后的那个坐标系，整体挪这么一段（米）。
# 两者合起来覆盖"尺寸"和"零点"两类最常见的实物误差。
#
# 最后一个关节的平移故意不给：它整段落在法兰系里，会被手眼解出来的 X 精确吸收，
# 拟合再多也定不下来，而且它和"爪尖偏移"本来就是同一件事（由「夹爪尖」那页单独标）。
#
# 没有修正文件时，这里所有函数都退化成"什么都不改"，工具的行为和以前完全一样。
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

# 关节数；本工具面向的都是六轴串联臂
JOINT_COUNT = 6
# 修正量的物理上限。超出这个范围说明拟合跑到荒唐的地方去了，宁可不用。
MAX_ZERO_DEG = 15.0
MAX_OFFSET_M = 0.05


def identity() -> dict:
    """什么都不改的修正。"""
    return {"zeros_deg": [0.0] * JOINT_COUNT,
            "offsets_m": [[0.0, 0.0, 0.0] for _ in range(JOINT_COUNT)],
            "source": "identity"}


def is_identity(correction) -> bool:
    """这份修正是不是等于没改。"""
    if not correction:
        return True
    zeros = np.asarray(correction.get("zeros_deg", [0.0] * JOINT_COUNT), dtype=float)
    offsets = np.asarray(correction.get("offsets_m", [[0.0] * 3] * JOINT_COUNT), dtype=float)
    return bool(np.allclose(zeros, 0.0) and np.allclose(offsets, 0.0))


def plausible(correction) -> tuple[bool, str]:
    """修正量是不是在物理上说得通的范围内。

    拟合是在"让板子更一致"这个目标下自由跑的，跑出来的数可能很小很合理，也可能
    跑到很远的地方换来一点点改善。落盘前必须过这一关。
    """
    zeros = np.asarray(correction.get("zeros_deg", []), dtype=float)
    offsets = np.asarray(correction.get("offsets_m", []), dtype=float)
    if zeros.size != JOINT_COUNT or offsets.shape != (JOINT_COUNT, 3):
        return False, "修正量的形状不对（%s / %s）" % (zeros.shape, offsets.shape)
    if np.abs(zeros).max() > MAX_ZERO_DEG:
        return False, "关节零点最大 %.1f°，超过 %.0f°" % (np.abs(zeros).max(), MAX_ZERO_DEG)
    if np.abs(offsets).max() > MAX_OFFSET_M:
        return False, "关节平移最大 %.0f mm，超过 %.0f mm" % (
            np.abs(offsets).max() * 1000, MAX_OFFSET_M * 1000)
    return True, ""


def load(path: Path | None = None) -> dict | None:
    """读修正产物；没有或读不动就返回 None（调用方按"不修正"处理）。"""
    if path is None or not Path(path).exists():
        return None
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    zeros = data.get("zeros_deg")
    offsets = data.get("offsets_m")
    if not isinstance(zeros, list) or len(zeros) != JOINT_COUNT:
        return None
    if not isinstance(offsets, list) or len(offsets) != JOINT_COUNT:
        return None
    return data


def save(correction: dict, path: Path, report: str = "") -> None:
    """落盘。写入失败直接抛——静默失败会让人以为修正生效了。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(correction)
    payload["zeros_deg"] = [float(v) for v in payload["zeros_deg"]]
    payload["offsets_m"] = [[float(v) for v in row] for row in payload["offsets_m"]]
    if report:
        payload["report"] = report
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def corrected_content(content: str, end_link: str, offsets_m) -> str:
    """按平移修正改一遍 URDF 文本；没有平移就原样返回。

    关节名要从标称链里取（URDF 里 joint 的书写顺序和关节链顺序不一定一致），
    所以这里先建一次标称链。
    """
    offsets = np.asarray(offsets_m, dtype=float)
    if offsets.size == 0 or np.allclose(offsets, 0.0):
        return content
    import re

    import kinpy

    stripped = re.sub(r"<\?xml[^?]*\?>", "", content, count=1)
    names = list(kinpy.build_serial_chain_from_urdf(stripped, end_link).get_joint_parameter_names())
    return _insert_offsets(stripped, names, offsets)


def corrected_joints(joints_deg, correction) -> np.ndarray:
    """把零点加回关节角：正运动学要走这个。"""
    joints = np.asarray(joints_deg, dtype=float)
    if correction is None:
        return joints
    zeros = np.asarray(correction.get("zeros_deg", [0.0] * JOINT_COUNT), dtype=float)
    return joints + zeros


def command_joints(joints_deg, correction) -> np.ndarray:
    """把修正模型的真实关节角换回臂接口使用的编码器角。"""
    joints = np.asarray(joints_deg, dtype=float)
    if correction is None:
        return joints
    zeros = np.asarray(correction.get("zeros_deg", [0.0] * JOINT_COUNT), dtype=float)
    return joints - zeros


def _insert_offsets(content: str, names: list, offsets: np.ndarray) -> str:
    """把每个关节的平移插进 URDF 的 origin 里。

    关节链是 ... origin_i · rot(q_i) · origin_{i+1} ...，而 origin_{i+1} 本来就是在
    第 i 个关节的子坐标系里表达的。所以"在第 i 个关节之后挪一段"就等于把 origin_{i+1}
    左乘上这段平移——不用另外造坐标系，改文本重新解析即可。
    最后一个关节的平移直接丢掉：它在法兰系里是常数，手眼解出来的 X 会把它吸收干净。
    """
    from xml.etree import ElementTree

    root = ElementTree.fromstring(content)
    by_name = {joint.get("name"): joint for joint in root.findall("joint")}
    for index, name in enumerate(names[:-1]):
        joint = by_name.get(name)
        if joint is None:
            continue
        origin = joint.find("origin")
        if origin is None:
            origin = ElementTree.SubElement(joint, "origin")
        shift = offsets[index]
        rpy = [float(v) for v in (origin.get("rpy") or "0 0 0").split()]
        xyz = np.array([float(v) for v in (origin.get("xyz") or "0 0 0").split()])
        xyz = xyz + R.from_euler("xyz", rpy).as_matrix() @ shift
        origin.set("xyz", " ".join("%.12g" % v for v in xyz))
    return ElementTree.tostring(root, encoding="unicode")


