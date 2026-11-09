# Description: 阻尼最小二乘（LM）逆运动学——模型无关的核心，谁用谁注入"误差"和"雅可比"。
#
# **真机和仿真共用这一份。** 两边的模型必须各用各的：仿真跑 Menagerie 的 MJCF，和部署
# URDF 差约 2°，用错模型那 2° 会混进结果里分不清是谁的错（见 sim/sim_ik.py 的说明）；
# 但求解算法、种子铺法、收敛判据、参数只有一处。
#
# 为什么要强调"一处"：这套算法原来只做在仿真那份（sim_ik.py）上，真机那份留在老的
# scipy SLSQP 上，一步动作光解 IK 就要一两秒（8~15 个起点 × 每个 ~150 ms），而且不会
# 报错——接口一样、结果也对，只是慢。同一批目标在真机链上量过（tools/ik_speed_benchmark.py
# 的另一种写法）：SLSQP 单起点 154 ms、12 个里 5 个到位；这套 LM 8 ms、12 个全到位。
import numpy as np

# 单次下降的迭代上限。**这个数是掐着失败代价定的**：到位的那一次中位只跑 7 轮、从零位
# 出发也常在 20 轮内收敛，而失败的下降在局部极小里能磨到上限。实测（工作区 120 个目标）：
# 上限 400→120 时"解不出来"的中位耗时 349 ms，压到 20 就只剩 96 ms，而到位率一个没掉
# （都是 117/120）；再往下压到 10，中位尝试次数从 1 涨到 3，常用档反而变慢。
IK_ITERS = 20
# 确定性起点颗数（Halton 序列）。铺满关节空间的起点换成低差异序列之后既可复现、又说不清
# "抽几颗才够"的问题也一并解决：到位率对颗数的实测拐点在 7~8 颗，这里备 16 颗，默认预算
# 用前几颗，调用方愿意多花时间还能往上加。
IK_SEEDS = 16
# 姿态误差折算成"米"的权重，把 6 维误差拼成一个标量代价。
IK_WEIGHT = 0.3
POS_TOL_M = 0.0005            # 到位判据：位置（米）
ROT_TOL_RAD = 0.005           # 到位判据：姿态（弧度）
DAMP_START = 0.02             # LM 初始阻尼：这一步接受就减半、被拒就乘三
DAMP_MIN, DAMP_MAX = 1e-6, 1e3
STEP_MAX_RAD = 0.5            # 单步关节增量上限：远起点别一步跨过头
STAGNATION_LIMIT = 8          # 连着这么多步几乎没有下降就认局部极小，收手
_HALTON_SEEDS: np.ndarray | None = None


def halton_seeds(count: int = IK_SEEDS) -> np.ndarray:
    """铺在单位立方体里的确定性起点（每维 0.15~0.85，离限位远一点），算一次缓存住。"""
    global _HALTON_SEEDS
    if _HALTON_SEEDS is None:
        primes, candidate = [], 2
        while len(primes) < 6:
            if all(candidate % p for p in range(2, int(candidate ** 0.5) + 1)):
                primes.append(candidate)
            candidate += 1
        points, index = [], 1
        while len(points) < IK_SEEDS:
            values = []
            for prime in primes:
                fraction, result, i = 1.0, 0.0, index
                while i > 0:
                    fraction /= prime
                    result += fraction * (i % prime)
                    i //= prime
                values.append(result)
            if all(0.15 < v < 0.85 for v in values):
                points.append(values)
            index += 1
        _HALTON_SEEDS = np.array(points)
    return _HALTON_SEEDS[:count]


def cost(error) -> float:
    """把一个 6 维误差 [位置(米), 姿态(弧度)] 折算成一个标量：位置 + 0.3×姿态。"""
    error = np.asarray(error, dtype=float)
    return float(np.linalg.norm(error[:3]) + IK_WEIGHT * np.linalg.norm(error[3:]))


def converged(error) -> bool:
    """位置与姿态都进了判据才算到位。"""
    error = np.asarray(error, dtype=float)
    return bool(np.linalg.norm(error[:3]) < POS_TOL_M
                and np.linalg.norm(error[3:]) < ROT_TOL_RAD)


def descend(error_at, jacobian_at, lo, hi, start):
    """从一个起点跌到最近的解，返回 (关节角, 最终误差)。

    LM：自适应阻尼 ＋ 这一步算完"代价确实下降"才接受 ＋ 顶到限位还往外顶的关节从这一步
    里摘掉。固定阻尼/固定步长的写法从远起点会在限位和局部极小之间空转（实测：同一批可达
    目标，从零位出发 28/40 到位，换这套是 117/120）。

    Args:
        error_at: `error_at(joints) -> 6 维误差`（位置米 + 姿态弧度，同一套坐标系）。
        jacobian_at: `jacobian_at(joints) -> 6×N 雅可比`，行序同误差、角速度行与误差的
            姿态部分同一套坐标（叉乘约定一致）。
        lo / hi: 关节限位（弧度）。
        start: 起点（关节角，弧度）。
    """
    joints = np.clip(np.asarray(start, dtype=float).copy(), lo, hi)
    error = np.asarray(error_at(joints), dtype=float)
    current_cost = float(error @ error)
    damping, stagnant = DAMP_START, 0
    for _ in range(IK_ITERS):
        if converged(error):
            break
        weighted = np.concatenate([error[:3], IK_WEIGHT * error[3:]])
        jacobian = np.array(jacobian_at(joints), dtype=float, copy=True)
        jacobian[3:, :] *= IK_WEIGHT
        accepted = False
        for _ in range(10):
            delta = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + damping * np.eye(6), weighted)
            blocked = (((joints <= lo + 1e-9) & (delta < 0.0))
                       | ((joints >= hi - 1e-9) & (delta > 0.0)))
            delta = np.where(blocked, 0.0, delta)
            length = float(np.linalg.norm(delta))
            if length > STEP_MAX_RAD:
                delta *= STEP_MAX_RAD / length
            trial = np.clip(joints + delta, lo, hi)
            trial_error = np.asarray(error_at(trial), dtype=float)
            trial_cost = float(trial_error @ trial_error)
            if trial_cost < current_cost:
                improved = current_cost - trial_cost
                joints, error, current_cost = trial, trial_error, trial_cost
                damping = max(damping * 0.5, DAMP_MIN)
                stagnant = stagnant + 1 if improved < 1e-10 else 0
                accepted = True
                break
            damping = min(damping * 3.0, DAMP_MAX)
        if not accepted or stagnant >= STAGNATION_LIMIT:
            break                       # 阻尼加到顶也降不下去 = 局部极小
    return joints, np.asarray(error_at(joints), dtype=float)


def multi_start(error_at, jacobian_at, lo, hi, start=None, budget: int = IK_SEEDS,
                accept=None) -> dict:
    """铺种子、逐颗下降；accept(关节角, 误差) 为真就收手。

    第一颗永远是 start（臂/模型现在的位置）：它到位就收手——既是"尽量别换分支"（关节空间
    走过去扫飞方块那几次的根因），也省掉其余尝试（实测：第一颗到位时，一次求解 476 ms →
    2.2 ms）。它没到位才往下铺确定性种子。

    Args:
        accept: 可选，"这一颗算数吗"的判据。默认"到位"（converged）；带闸的调用方传
            "到位 且 关节没走超过闸"。

    Returns:
        {"joints", "error", "attempts"}；attempts 是每颗种子的记录（谁被采用、各试了
        什么），调用方拿它落 debug。全都算数不了时 joints/error 是**误差最小**的那颗，
        由调用方按误差判——"交回一个解"不等于"到位"。
    """
    accept = accept or (lambda joints, error: converged(error))
    seeds = []
    if start is not None:
        seeds.append(np.asarray(start, dtype=float).reshape(len(lo)))
    for unit in halton_seeds(max(0, int(budget) - len(seeds))):
        seeds.append(lo + np.asarray(unit, dtype=float) * (hi - lo))
    attempts, best = [], None
    for index, seed in enumerate(seeds):
        joints, error = descend(error_at, jacobian_at, lo, hi, seed)
        accepted = bool(accept(joints, error))
        attempts.append({"index": index, "joints": joints, "error": error, "accepted": accepted})
        candidate = (cost(error), joints, error)
        if best is None or candidate[0] < best[0]:
            best = candidate
        if accepted:
            break
    _, joints, error = best
    return {"joints": joints, "error": error, "attempts": attempts}
