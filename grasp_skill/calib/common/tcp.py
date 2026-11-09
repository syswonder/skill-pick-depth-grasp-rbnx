# Description: 标定夹爪尖在法兰坐标系里的位置（TCP），用"同一个点、不同姿态"的转轴法。
#
# 为什么需要这一步：平面标定不需要它。平面标定把最后一个轴保持垂直向下，夹爪尖就在法兰正
# 下方，尖端相对法兰的偏移只影响 z，而 z 已经被桌面高度那一步吸成一个常数了，所以那套流程
# 里夹爪有多长根本没出现过。三维标定的接近方向是任意的，这段偏移会转到 x 和 y 上去，不再是
# 常数，必须单独量出来。
#
# 怎么量：把一个尖头物件固定在臂能碰到的地方（螺丝尖、钉子都行），拖动示教让夹爪尖对准它，
# 换几个明显不同的手腕姿态各记一次。物件没动过，所以每个位姿都满足
#     R_i · p_tip + t_i = q
# 其中 p_tip 是夹爪尖在法兰系里的位置（要求的），q 是那个点在基座系里的位置（不需要知道）。
# 两个位姿相减，q 就消掉了：
#     (R_i - R_j) · p_tip = t_j - t_i
# 所以用户不需要量那个点在哪儿，只要让它别动。这跟手眼标定不需要知道标定板放在哪儿是同一
# 个道理。
#
# 解得出来的条件：把各对位姿的 (R_i - R_j) 摞起来，秩必须是 3。手腕姿态换得不够开时秩不足，
# 缺的那个方向上夹爪尖的位置解不出来，判据（observability）会拦住。
import numpy as np

from common.handeye3d import rotation_angle_deg, split_transform

# URDF 里 joint7/joint8 的原点在 link6 的 +z 方向 0.1358 m 处，手指尖还要再往外一点。
# 这个值只用来跟标出来的结果对一下，不参与求解。
NOMINAL_TIP_FLANGE = np.array([0.0, 0.0, 0.1358])


def solve(flange_transforms: list) -> dict:
    """解夹爪尖在法兰系里的位置。

    Args:
        flange_transforms: 每次碰到同一个点时，法兰在基座系的 4x4 位姿（正运动学算出来的）。

    Returns:
        {"tip_in_flange": 3 维，米；"point_in_base": 3 维，米；
         "residuals_m": 每个位姿的偏差；"rms_m"；"max_m"；"n"}

    Raises:
        ValueError: 位姿少于 3 个。
    """
    if len(flange_transforms) < 3:
        raise ValueError(f"位姿只有 {len(flange_transforms)} 个，少于 3 个解不出夹爪尖位置")
    rows, rhs = [], []
    for transform in flange_transforms:
        rotation, translation = split_transform(transform)
        rows.append(np.hstack([rotation, -np.eye(3)]))
        rhs.append(-translation)
    matrix = np.vstack(rows)
    vector = np.concatenate(rhs)
    solution, *_ = np.linalg.lstsq(matrix, vector, rcond=None)
    tip_in_flange, point_in_base = solution[:3], solution[3:]
    residuals = []
    for transform in flange_transforms:
        rotation, translation = split_transform(transform)
        residuals.append(float(np.linalg.norm(rotation @ tip_in_flange + translation - point_in_base)))
    residuals = np.array(residuals)
    return {
        "tip_in_flange": tip_in_flange,
        "point_in_base": point_in_base,
        "residuals_m": residuals,
        "rms_m": float(np.sqrt(np.mean(residuals ** 2))),
        "max_m": float(residuals.max()),
        "n": len(flange_transforms),
    }


def predicted_error_m(sigma_min: float, touch_rms_m: float, n_poses: int) -> float:
    """按条件数估一下夹爪尖位置能标到多准，单位米。

    最小二乘 M·p = b 的解误差有界：||dp|| <= ||db|| / sigma_min。每次碰点带三个方向的对准
    误差，n 个位姿就是 sqrt(3n) 倍，所以 ||db|| = touch_rms * sqrt(3n)。

    这个估计比实测的中位误差保守约三成（仿真扫出来的比值稳定在 0.75 上下），拿它当上限用。
    实测最差误差大约是中位的两到三倍。
    """
    if sigma_min <= 1e-12:
        return float("inf")
    return float(touch_rms_m * np.sqrt(3.0 * n_poses) / sigma_min)


def observability(flange_transforms: list, min_angle_deg: float = 10.0,
                  sigma_min_required: float = 2.0) -> dict:
    """量这批位姿够不够解出夹爪尖的位置。

    对每一对位姿求 (R_i - R_j)，摞成 3n x 3 的矩阵。它的最小奇异值 sigma_min 就是放大倍数的
    倒数：碰点时每偏 1 mm，解出来的夹爪尖位置最多偏 1/sigma_min mm。手腕姿态换得不够开时
    sigma_min 接近零，缺的方向由 weak_direction 指出来。

    合格线为什么定在 2.0：仿真扫过手腕摆动幅度与实际误差的关系（8 次碰点、1 mm 对准误差），
    摆动 10° 时 sigma_min 约 0.8，误差最差 12.7 mm；摆动 30° 时 sigma_min 约 2.5，误差最差
    4.3 mm。要让最差误差落在 5 mm 以内，sigma_min 得到 2 以上，对应手腕至少摆开正负 30°。

    这一条判据是必须的，不能用拟合残差代替：残差在上面那组扫描里始终是 2.2 mm 左右，摆动
    2° 时真实误差已经 21 mm 了，残差还是 2.2 mm。残差看不出姿态换得够不够开。

    Args:
        flange_transforms: 各次碰点时法兰在基座系的位姿。
        min_angle_deg: 两个位姿的姿态差小于它就不算一对，姿态没换的一对提供不了信息。
        sigma_min_required: 合格线。

    Returns:
        判据字典。
    """
    rotations = [split_transform(transform)[0] for transform in flange_transforms]
    differences, angles = [], []
    for i in range(len(rotations)):
        for j in range(i + 1, len(rotations)):
            angle = rotation_angle_deg(rotations[i].T @ rotations[j])
            if angle < min_angle_deg:
                continue
            differences.append(rotations[i] - rotations[j])
            angles.append(angle)
    result = {"pairs": len(differences), "min_angle_deg": min_angle_deg}
    if len(differences) < 2:
        result.update({"ok": False, "why": f"只有 {len(differences)} 对位姿的姿态差超过 "
                                           f"{min_angle_deg}°，至少要 2 对。换几个明显不同的"
                                           "手腕姿态再碰几次，别每次都用同一个姿态。"})
        return result
    stacked = np.vstack(differences)
    _, singulars, vt = np.linalg.svd(stacked)
    sigma_min = float(singulars[-1])
    result.update({
        "singulars": singulars,
        "sigma_min": sigma_min,
        "gain": float(1.0 / sigma_min) if sigma_min > 1e-12 else float("inf"),
        "weak_direction": vt[-1],
        "max_angle_deg": float(max(angles)),
        "ok": bool(sigma_min >= sigma_min_required),
    })
    if not result["ok"]:
        weak = vt[-1]
        result["why"] = (f"手腕姿态换得不够开（sigma_min {sigma_min:.3f} < {sigma_min_required}），"
                        f"沿法兰系方向 [{weak[0]:.2f}, {weak[1]:.2f}, {weak[2]:.2f}] 的信息不足，"
                        "夹爪尖在这个方向上的位置解不出来。手腕至少要摆开正负 30°，"
                        "而且要绕不同的轴摆，别只把同一根轴来回转。")
    return result


def format_report(result: dict, checks: dict, nominal=NOMINAL_TIP_FLANGE,
                  budget_mm: float = 5.0, touch_rms_m: float = 0.001) -> str:
    """把 TCP 标定的结果与判据写成一份能读的报告。"""
    tip = result["tip_in_flange"]
    lines = [
        f"夹爪尖标定报告（{result['n']} 次碰点）",
        "",
        f"夹爪尖在法兰坐标系里的位置（毫米）："
        f"x {tip[0] * 1000:.1f}, y {tip[1] * 1000:.1f}, z {tip[2] * 1000:.1f}",
        f"离法兰 {np.linalg.norm(tip) * 1000:.1f} mm",
        "",
        "一、这批位姿够不够解",
        f"  参与判据的位姿对：{checks['pairs']} 对（姿态差超过 {checks['min_angle_deg']}° 的才算）",
    ]
    if "sigma_min" in checks:
        # 精度上限要拿实测的碰点离散度去算，不能拿一个假设的常数：
        # 对准误差多大是这台臂 + 这个碰点方式决定的，只有数据知道。
        measured = float(result["rms_m"])
        used = max(touch_rms_m, measured)
        predicted = predicted_error_m(checks["sigma_min"], used, result["n"])
        lines += [
            f"  最小奇异值：{checks['sigma_min']:.3f}（要 >= 2.0），"
            f"碰点误差放大 {checks['gain']:.2f} 倍",
            f"  最大姿态差：{checks['max_angle_deg']:.1f}°",
            f"  按条件数估的精度上限：{predicted * 1000:.2f} mm"
            f"（每次碰点的离散度按实测 {used * 1000:.1f} mm 算）",
        ]
    lines.append(f"  结论：{'可解' if checks.get('ok') else '解不出来'}")
    if checks.get("why"):
        lines.append(f"  原因：{checks['why']}")
    lines += [
        "",
        "二、碰点重复性（同一个点在基座系里算出来的位置，各次之间差多少）",
        f"  均方根 {result['rms_m'] * 1000:.2f} mm，最大 {result['max_m'] * 1000:.2f} mm",
        "  这个数就是每次碰点对准得准不准，它是上面那条精度估计的输入——"
        "两个数要一起看：这条判据决定姿态够不够张开（位置有没有被唯一确定），"
        "这个残差决定每次碰得准不准（那个唯一解准不准）。",
    ]
    if result["rms_m"] > touch_rms_m * 3:
        lines.append("  残差偏大：每次碰点前后差了这么多，说明碰的方法本身重复不下来。"
                     "钝的爪尖对着一个小点硬碰最多只能到这个量级；"
                     "换一个能卡住爪尖的目标（锥形沉孔、V 形槽）比练手感有效。")
    worst = int(np.argmax(result["residuals_m"]))
    lines.append(f"  偏得最多的是第 {worst + 1} 次：{result['residuals_m'][worst] * 1000:.2f} mm")
    if nominal is not None:
        difference = np.linalg.norm(tip - np.asarray(nominal, dtype=float)) * 1000
        lines += [
            "",
            "三、与 URDF 标称值对比",
            f"  URDF 标称（毫米）：x {nominal[0] * 1000:.1f}, y {nominal[1] * 1000:.1f}, "
            f"z {nominal[2] * 1000:.1f}",
            f"  差 {difference:.1f} mm",
        ]
        if difference > 15.0:
            lines.append("  差得比较多。先确认碰点时对准的是同一个位置，再确认夹爪支架是不是"
                         "跟 URDF 里画的一样；支架换过或者加了垫片，这个差值就是真的。")
    return "\n".join(lines)
