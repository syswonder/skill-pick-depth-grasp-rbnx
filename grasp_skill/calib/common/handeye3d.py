# Description: 眼在手上的三维手眼标定：解 AX=XB，并把"这组位姿够不够把 X 解出来"量成数。
#
# 与平面标定的区别：common/homography.py 解的是一张图像到桌面的单应，物体必须贴在桌面上，
# 相机也必须固定不动。这里解的是相机相对夹爪的固定位姿 X = T_gripper_camera，有了它，
# 深度相机看到的任意三维点都能换到臂基座系，物体不必贴桌面，相机也可以跟着臂一起动。
#
# 解得出来的条件（observability() 把这两条量成数）：
#   转动部分：只要有两次相对运动的转轴不平行，X 的旋转就唯一确定。
#   平移部分：把各次相对运动的 (R_A - I) 摞成一个 3n x 3 的矩阵，它的秩必须是 3。
#            所有转动都绕同一根轴时秩只有 2，零空间正好是那根轴，X 的平移沿该轴解不出来。
#
# 本文件只做数学，不碰硬件、不读配置：输入是若干对（夹爪在基座系的位姿，标定板在相机系的
# 位姿），输出是 X 与一份能读的报告。
import cv2
import numpy as np

# 几种解法都试一遍，互相对不上就说明输入有问题（见 cross_method_spread）。
#
# 不用 ANDREFF：它把 AX=XB 当线性最小二乘解，不强制旋转部分保持正交，有噪声时会给出
# 偏掉的平移。实测同一批干净数据，其余四种互相只差 0.8 mm，它一个人偏 28.5 mm，
# 留在列表里会把正常结果写成"输入有问题"。
METHODS = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def make_transform(rotation, translation) -> np.ndarray:
    """把 3x3 旋转与 3 维平移拼成 4x4 齐次矩阵。"""
    transform = np.eye(4)
    transform[:3, :3] = np.asarray(rotation, dtype=float).reshape(3, 3)
    transform[:3, 3] = np.asarray(translation, dtype=float).reshape(3)
    return transform


def split_transform(transform) -> tuple[np.ndarray, np.ndarray]:
    """4x4 齐次矩阵拆回 (3x3 旋转, 3 维平移)。"""
    transform = np.asarray(transform, dtype=float)
    return transform[:3, :3].copy(), transform[:3, 3].copy()


def rotation_angle_deg(rotation) -> float:
    """旋转矩阵对应的转角，单位度。"""
    cosine = (np.trace(np.asarray(rotation, dtype=float)[:3, :3]) - 1.0) / 2.0
    return float(np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0))))


def rotation_axis_angle(rotation) -> tuple[np.ndarray | None, float]:
    """旋转矩阵的转轴（单位向量）与转角（弧度）；转角太小时转轴返回 None。"""
    rodrigues, _ = cv2.Rodrigues(np.asarray(rotation, dtype=float)[:3, :3])
    angle_rad = float(np.linalg.norm(rodrigues))
    if angle_rad < 1e-9:
        return None, 0.0
    return rodrigues.reshape(3) / angle_rad, angle_rad


def solve(gripper_to_base: list, target_to_cam: list, method: str = "PARK") -> np.ndarray:
    """解 X = T_gripper_camera，也就是相机在夹爪坐标系里的固定位姿。

    Args:
        gripper_to_base: 每个位姿下夹爪在臂基座系的 4x4 位姿，由关节角做正运动学得到。
        target_to_cam: 同一批位姿下标定板在相机系的 4x4 位姿，由 solvePnP 得到。
        method: METHODS 里的解法名。

    Returns:
        4x4 的 X。

    Raises:
        ValueError: 解法名不认识、两边数量不一致，或位姿少于 3 组。
        RuntimeError: OpenCV 求解失败。
    """
    if method not in METHODS:
        raise ValueError(f"不认识的解法 {method}，可选：{list(METHODS)}")
    if len(gripper_to_base) != len(target_to_cam):
        raise ValueError(f"两边位姿数量不一致：夹爪 {len(gripper_to_base)} 个，标定板 {len(target_to_cam)} 个")
    if len(gripper_to_base) < 3:
        raise ValueError(f"位姿只有 {len(gripper_to_base)} 个，少于 3 个解不出 X")
    pairs_g = [split_transform(transform) for transform in gripper_to_base]
    pairs_c = [split_transform(transform) for transform in target_to_cam]
    rotation, translation = cv2.calibrateHandEye(
        [R for R, _ in pairs_g], [t.reshape(3, 1) for _, t in pairs_g],
        [R for R, _ in pairs_c], [t.reshape(3, 1) for _, t in pairs_c],
        method=METHODS[method])
    if rotation is None or translation is None:
        raise RuntimeError("cv2.calibrateHandEye 没解出结果")
    return make_transform(rotation, translation)


def board_in_base(X, gripper_to_base: list, target_to_cam: list) -> list:
    """每个位姿各自算一遍标定板在基座系的位姿：T_base_board = T_base_gripper · X · T_cam_board。

    标定板在整个采集过程里没动过，所以这 N 个结果本该完全一样。它们的离散程度就是这套标定
    的真实误差，单位是米和度。这个数不需要另外量任何东西，也不需要知道标定板放在哪里。
    """
    return [np.asarray(T_bg, dtype=float) @ np.asarray(X, dtype=float) @ np.asarray(T_cb, dtype=float)
            for T_bg, T_cb in zip(gripper_to_base, target_to_cam)]


def _mean_rotation(rotations: list) -> np.ndarray:
    """一组旋转矩阵的平均：3x3 求和之后用 SVD 投影回 SO(3)。"""
    total = np.sum([np.asarray(rotation, dtype=float) for rotation in rotations], axis=0)
    u, _, vt = np.linalg.svd(total)
    if np.linalg.det(u @ vt) < 0:
        u[:, -1] *= -1
    return u @ vt


def consistency(X, gripper_to_base: list, target_to_cam: list) -> dict:
    """标定板位姿的一致性：位置散布（米）与朝向散布（度）。"""
    boards = board_in_base(X, gripper_to_base, target_to_cam)
    origins = np.array([transform[:3, 3] for transform in boards])
    center = origins.mean(axis=0)
    position_errors = np.linalg.norm(origins - center, axis=1)
    mean_rotation = _mean_rotation([transform[:3, :3] for transform in boards])
    angle_errors = np.array([rotation_angle_deg(mean_rotation.T @ transform[:3, :3]) for transform in boards])
    return {
        "n": len(boards),
        "board_center_m": center,
        "board_rotation": mean_rotation,
        "position_rms_m": float(np.sqrt(np.mean(position_errors ** 2))),
        "position_max_m": float(position_errors.max()),
        "angle_rms_deg": float(np.sqrt(np.mean(angle_errors ** 2))),
        "angle_max_deg": float(angle_errors.max()),
        "per_pose_position_m": position_errors,
    }


def _max_axis_angle_deg(axes: list) -> float:
    """一组转轴里两两之间最大的夹角，单位度。轴的正负号没有意义，所以只看到 90°。"""
    best = 0.0
    for i in range(len(axes)):
        for j in range(i + 1, len(axes)):
            cosine = abs(float(np.dot(axes[i], axes[j])))
            best = max(best, float(np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0)))))
    return best


def observability(gripper_to_base: list, min_angle_deg: float = 5.0,
                  axis_spread_min: float = 0.25, translation_sigma_min: float = 0.5) -> dict:
    """量这组位姿够不够把 X 解出来，以及噪声会被放大多少倍。

    对每一对位姿 (i, j) 取夹爪的相对运动 A = T_i^-1 · T_j：
      转动部分：A 的转轴张得越开，X 的旋转解得越稳。所有轴平行时旋转解不出来。
      平移部分：X 的平移满足 (R_A - I) t_x = R_x t_B - t_A。把所有 (R_A - I) 摞成 3n x 3
                的矩阵解最小二乘，它的最小奇异值 sigma_min 就是放大倍数的倒数：右端每有
                1 mm 噪声，t_x 最多偏 1/sigma_min mm。sigma_min 为零就是解不出来。

    Args:
        gripper_to_base: 各位姿下夹爪在基座系的 4x4 位姿。
        min_angle_deg: 相对转角小于它的位姿对不参与判据，转得太少的一对提供不了信息。
        axis_spread_min: 转轴散布的合格线，第二大奇异值与最大奇异值的比值。
        translation_sigma_min: 平移最小奇异值的合格线。

    Returns:
        判据字典，format_report() 负责把它写成人能读的话。
    """
    transforms = [np.asarray(transform, dtype=float) for transform in gripper_to_base]
    axes, delta_rotations = [], []
    for i in range(len(transforms)):
        for j in range(i + 1, len(transforms)):
            relative = np.linalg.inv(transforms[i]) @ transforms[j]
            axis, angle_rad = rotation_axis_angle(relative)
            if axis is None or np.rad2deg(angle_rad) < min_angle_deg:
                continue
            axes.append(axis)
            delta_rotations.append(relative[:3, :3] - np.eye(3))
    result = {"pairs": len(axes), "min_angle_deg": min_angle_deg}
    if len(axes) < 2:
        result.update({"rotation_ok": False, "translation_ok": False,
                       "why": f"只有 {len(axes)} 对位姿的相对转角超过 {min_angle_deg}°，至少要 2 对。"
                              "把手腕转起来，别只让臂平移。"})
        return result
    axis_singulars = np.linalg.svd(np.array(axes), compute_uv=False)
    stacked = np.vstack(delta_rotations)
    _, translation_singulars, vt = np.linalg.svd(stacked)
    axis_spread = float(axis_singulars[1] / axis_singulars[0])
    sigma_min = float(translation_singulars[-1])
    result.update({
        "axis_singulars": axis_singulars,
        "axis_spread": axis_spread,
        "max_axis_angle_deg": _max_axis_angle_deg(axes),
        "translation_singulars": translation_singulars,
        "translation_sigma_min": sigma_min,
        "translation_gain": float(1.0 / sigma_min) if sigma_min > 1e-12 else float("inf"),
        "translation_weak_direction": vt[-1],
        "rotation_ok": bool(axis_spread >= axis_spread_min),
        "translation_ok": bool(sigma_min >= translation_sigma_min),
    })
    if not result["rotation_ok"]:
        result["why"] = (f"各次转动的轴几乎都平行（散布 {axis_spread:.3f} < {axis_spread_min}），"
                        "X 的旋转解不唯一。换两个方向的手腕姿态再采几张。")
    elif not result["translation_ok"]:
        weak = vt[-1]
        result["why"] = (f"沿夹爪系方向 [{weak[0]:.2f}, {weak[1]:.2f}, {weak[2]:.2f}] 的平移几乎没有"
                        f"信息（sigma_min {sigma_min:.3f} < {translation_sigma_min}），"
                        "这个方向上相机装在哪儿解不出来。绕另一根轴转一转再采几张。")
    return result


def leave_one_out(gripper_to_base: list, target_to_cam: list, method: str = "PARK") -> dict:
    """每次去掉一个位姿重解一遍 X，看结果晃多少。晃得大说明有位姿拖了后腿。"""
    if len(gripper_to_base) < 5:
        return {"n": 0}
    origins, rotations = [], []
    for skip in range(len(gripper_to_base)):
        keep_gripper = [t for i, t in enumerate(gripper_to_base) if i != skip]
        keep_board = [t for i, t in enumerate(target_to_cam) if i != skip]
        try:
            X = solve(keep_gripper, keep_board, method=method)
        except (ValueError, RuntimeError, cv2.error):
            continue
        origins.append(X[:3, 3])
        rotations.append(X[:3, :3])
    if len(origins) < 2:
        return {"n": len(origins)}
    origins = np.array(origins)
    mean_rotation = _mean_rotation(rotations)
    return {
        "n": len(origins),
        "position_spread_m": float(np.linalg.norm(origins - origins.mean(axis=0), axis=1).max()),
        "angle_spread_deg": float(max(rotation_angle_deg(mean_rotation.T @ r) for r in rotations)),
    }


def cross_method_spread(gripper_to_base: list, target_to_cam: list) -> dict:
    """几种解法各解一遍，对不上就说明输入有问题。

    判据要能分清是数据有问题还是某一个解法自己有问题：有一种解法在数据正常时也会单独
    跑偏二十几毫米，历史上就是这么把一份正常报告写成"输入有问题"的。把最不合群的那个
    单独摘出来，剩下几种之间的差才是判据。
    """
    solved = {}
    for name in METHODS:
        try:
            solved[name] = solve(gripper_to_base, target_to_cam, method=name)
        except (ValueError, RuntimeError, cv2.error):
            continue
    names = list(solved)
    if len(names) < 2:
        return {"solved": names, "position_spread_m": None, "angle_spread_deg": None}
    origins = np.array([solved[n][:3, 3] for n in names])
    # 每个解法到其余解法的中位距离，最不合群的那个这个数会明显更大
    medians = np.array([
        float(np.median([np.linalg.norm(origins[i] - origins[j])
                         for j in range(len(names)) if j != i]))
        for i in range(len(names))])
    order = np.argsort(medians)
    outlier, keep = None, list(range(len(names)))
    # 要同时满足"比第二不合群的明显大一截"和"自己差得够多"：只看倍数的话，几个解法都
    # 只差零点几毫米、其中一个差三毫米，也会被当成异常摘出去，而那三毫米可能是有意义的。
    if len(names) >= 4 and medians[order[-1]] > max(4.0 * medians[order[-2]], 0.010):
        outlier = names[order[-1]]
        keep = [i for i in keep if i != order[-1]]
    kept = origins[keep]
    mean_rotation = _mean_rotation([solved[names[i]][:3, :3] for i in keep])
    return {
        "solved": names,
        "agreed": len(keep),
        "outlier": outlier,
        "outlier_offset_m": float(medians[order[-1]]) if outlier else None,
        "position_spread_m": float(np.linalg.norm(kept - kept.mean(axis=0), axis=1).max()),
        "angle_spread_deg": float(max(rotation_angle_deg(mean_rotation.T @ solved[names[i]][:3, :3])
                                      for i in keep)),
        "transforms": solved,
    }


def format_report(X, gripper_to_base: list, target_to_cam: list, method: str = "PARK",
                  budget_mm: float = 5.0) -> str:
    """把一次手眼标定的结果与全部判据写成一份能读的报告。

    Args:
        X: solve() 解出的 4x4 相机位姿。
        gripper_to_base: 各位姿下夹爪在基座系的位姿。
        target_to_cam: 各位姿下标定板在相机系的位姿。
        method: 得到 X 用的解法名，只用于写进报告。
        budget_mm: 抓取允许的误差预算，毫米。报告拿标定板散布跟它比。

    Returns:
        多行文本。
    """
    checks = observability(gripper_to_base)
    spread = consistency(X, gripper_to_base, target_to_cam)
    loo = leave_one_out(gripper_to_base, target_to_cam, method=method)
    across = cross_method_spread(gripper_to_base, target_to_cam)
    lines = [
        f"手眼标定报告（解法 {method}，{len(gripper_to_base)} 个位姿）",
        "",
        "相机在夹爪坐标系里的位置 t（毫米）："
        f" x {X[0, 3] * 1000:.1f}, y {X[1, 3] * 1000:.1f}, z {X[2, 3] * 1000:.1f}",
        f"相机相对夹爪转过的角度：{rotation_angle_deg(X):.2f}°",
        "",
        "一、这组位姿够不够解（数学可解性）",
        f"  参与判据的位姿对：{checks['pairs']} 对（相对转角超过 {checks['min_angle_deg']}° 的才算）",
    ]
    if "axis_spread" in checks:
        lines += [
            f"  转轴散布：{checks['axis_spread']:.3f}（要 >= 0.25），"
            f"最大两轴夹角 {checks['max_axis_angle_deg']:.1f}°",
            f"  平移最小奇异值：{checks['translation_sigma_min']:.3f}（要 >= 0.5），"
            f"噪声放大 {checks['translation_gain']:.2f} 倍",
        ]
    lines += [
        f"  旋转部分：{'可解' if checks.get('rotation_ok') else '解不出来'}",
        f"  平移部分：{'可解' if checks.get('translation_ok') else '解不出来'}",
    ]
    if checks.get("why"):
        lines.append(f"  原因：{checks['why']}")
    lines += [
        "",
        "二、实际精度（标定板在基座系的位姿本该不动，这里量它晃了多少）",
        f"  位置：均方根 {spread['position_rms_m'] * 1000:.2f} mm，最大 {spread['position_max_m'] * 1000:.2f} mm",
        f"  朝向：均方根 {spread['angle_rms_deg']:.3f}°，最大 {spread['angle_max_deg']:.3f}°",
        f"  误差预算 {budget_mm:.1f} mm："
        f"{'够用' if spread['position_max_m'] * 1000 <= budget_mm else '超了，见下面的逐位姿偏差'}",
    ]
    worst = np.argsort(spread["per_pose_position_m"])[::-1][:3]
    lines.append("  偏得最多的三个位姿："
                 + "，".join(f"第 {i + 1} 个 {spread['per_pose_position_m'][i] * 1000:.2f} mm" for i in worst))
    lines += ["", "三、稳定性交叉检查"]
    if loo.get("n"):
        lines.append(f"  去掉任一位姿重解（{loo['n']} 次）：位置最多变 {loo['position_spread_m'] * 1000:.2f} mm，"
                     f"角度最多变 {loo['angle_spread_deg']:.3f}°")
    else:
        lines.append("  位姿不足 5 个，跳过去一留一检查")
    if across.get("position_spread_m") is not None:
        if across.get("outlier"):
            lines.append(f"  其余 {across['agreed']} 种解法之间：位置差 "
                         f"{across['position_spread_m'] * 1000:.2f} mm，"
                         f"角度差 {across['angle_spread_deg']:.3f}°")
            lines.append(f"  {across['outlier']} 单独跟它们差 {across['outlier_offset_m'] * 1000:.2f} mm："
                         f"差这么多通常出在它自己身上，不计入判据")
        else:
            lines.append(f"  {len(across['solved'])} 种解法之间：位置差 "
                         f"{across['position_spread_m'] * 1000:.2f} mm，"
                         f"角度差 {across['angle_spread_deg']:.3f}°")
    lines += ["", f"标定板在基座系的位置（毫米）：x {spread['board_center_m'][0] * 1000:.1f}, "
                  f"y {spread['board_center_m'][1] * 1000:.1f}, z {spread['board_center_m'][2] * 1000:.1f}"]
    return "\n".join(lines)
