# Description: 图像平面 ↔ 机械臂平面之间的纯数学（2D 手眼标定用的单应变换）。
#
# 标定工具里只有这一份，两个部署都用它——差别只在 RANSAC 判据：
#   S10 以像素为单位给 5.0（默认值就是它历史上用的那组）
#   piper 历史上以米为单位给 0.005、迭代更多、置信度更高（传参覆盖）
import cv2
import numpy as np


def image_to_robot_xy(homography_matrix: np.ndarray, image_point) -> np.ndarray:
    """把图像像素点映射到机械臂平面的 (x, y)。

    Args:
        homography_matrix: 3x3 单应性矩阵（图像平面 → 机械臂平面）。
        image_point: 像素坐标，形如 (u, v)。

    Returns:
        机械臂基座系下的 (x, y)，单位为米。
    """
    image_point_homogeneous = np.array([image_point[0], image_point[1], 1.0], dtype=np.float64)
    robot_point_homogeneous = homography_matrix @ image_point_homogeneous
    return robot_point_homogeneous[:2] / robot_point_homogeneous[2]


def calibrate_2d_from_correspondences(
    image_points: np.ndarray,
    robot_points: np.ndarray,
    ransac_threshold: float = 5.0,
    max_iters: int = 2000,
    confidence: float = 0.995,
) -> tuple[np.ndarray, np.ndarray]:
    """由成对的图像点/机械臂点拟合单应性矩阵（RANSAC）。

    Args:
        image_points: Nx2 像素坐标。
        robot_points: Nx2 机械臂平面坐标，单位为米，与 image_points 一一对应。
        ransac_threshold: RANSAC 内点判据；单位随传进来的点走（像素或米）。
        max_iters: RANSAC 最大迭代次数。
        confidence: RANSAC 置信度。

    Returns:
        (3x3 单应性矩阵 float32, 长度为 N 的内点掩码 bool)。

    Raises:
        ValueError: 点的形状不对，或两边的数量不一致，或不足 4 对点。
        RuntimeError: OpenCV 求解失败。
    """
    image_points = np.asarray(image_points, dtype=np.float32)
    robot_points = np.asarray(robot_points, dtype=np.float32)

    if image_points.ndim != 2 or image_points.shape[1] != 2:
        raise ValueError("图片点数据格式错误，应该是Nx2")
    if robot_points.ndim != 2 or robot_points.shape[1] != 2:
        raise ValueError("机械臂平面点数据格式错误，应该是Nx2")
    if image_points.shape[0] != robot_points.shape[0]:
        raise ValueError("图片点和机械臂平面点数量不匹配")
    if image_points.shape[0] < 4:
        raise ValueError("标定点数量不足，至少需要4个点")

    homography_matrix, inlier_mask = cv2.findHomography(
        image_points,
        robot_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=float(ransac_threshold),
        maxIters=int(max_iters),
        confidence=float(confidence),
    )
    if homography_matrix is None or inlier_mask is None:
        raise RuntimeError("计算单应性矩阵失败")
    return homography_matrix.astype(np.float32), inlier_mask.reshape(-1).astype(bool)


def compute_reprojection_errors(homography_matrix: np.ndarray, image_points: np.ndarray, robot_points: np.ndarray) -> np.ndarray:
    """逐点重投影误差：图像点映射到臂平面后，与实测臂坐标的欧氏距离（米）。"""
    projected_points = np.array(
        [image_to_robot_xy(homography_matrix, point) for point in image_points],
        dtype=np.float32,
    )
    return np.linalg.norm(projected_points - robot_points, axis=1)


def format_report(homography_matrix: np.ndarray, image_points: np.ndarray, robot_points: np.ndarray,
                  inlier_mask: np.ndarray) -> str:
    """把拟合结果写成一段人能读的报告（两边共用同一种格式）。"""
    errors = compute_reprojection_errors(homography_matrix, image_points, robot_points)
    inlier_errors = errors[inlier_mask]
    lines = [
        f"总点数: {len(image_points)}",
        f"RANSAC内点数: {int(inlier_mask.sum())}",
        f"全部点平均误差: {float(errors.mean()):.6f} m / 最大 {float(errors.max()):.6f} m",
    ]
    if len(inlier_errors) > 0:
        lines.append(f"内点平均误差: {float(inlier_errors.mean()):.6f} m"
                     f" / 最大 {float(inlier_errors.max()):.6f} m")
    lines.append("单应性矩阵:")
    lines.append(str(homography_matrix))
    return "\n".join(lines)
