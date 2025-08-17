import time
import random
import statistics
import math
from dataclasses import dataclass
from typing import List, Tuple
from tqdm import tqdm

# ========== 可调参数 ==========
RNG_SEED = 20250816
PROFILE_SAMPLES_PER_CLASS = 200   # 画像阶段：每类采样次数
INFER_SAMPLES = 500               # 推断阶段：未知标签请求数量
NETWORK_JITTER_MEAN = 0.0         # 额外网络抖动（均值，秒）
NETWORK_JITTER_STD  = 0.005       # 额外网络抖动（标准差，秒），默认5ms
SUCCESS_BASE = 0.050              # 成功路径内部额外耗时，秒（50ms）
FAIL_BASE    = 0.020              # 失败路径内部额外耗时，秒（20ms）
RANDOM_DELAY_MIN = 0.100          # 随机延迟最小值，秒（100ms）
RANDOM_DELAY_MAX = 0.300          # 随机延迟最大值，秒（300ms）
FIXED_TARGET_TIME = 0.300         # 固定延迟目标总耗时，秒（300ms）
# ============================

random.seed(RNG_SEED)

@dataclass
class AuthServer:
    mode: str  # "random_delay" | "fixed_delay"

    def one_request(self, will_success: bool) -> float:
        """
        模拟一次认证请求，返回总耗时（秒）
        """
        start = time.perf_counter()

        # 内部不同分支的基础耗时（模拟密码校验 / 早返回等）
        if will_success:
            time.sleep(SUCCESS_BASE)
        else:
            time.sleep(FAIL_BASE)

        if self.mode == "random_delay":
            # 注入随机延迟
            time.sleep(random.uniform(RANDOM_DELAY_MIN, RANDOM_DELAY_MAX))
        elif self.mode == "fixed_delay":
            # 统一响应时间：补齐到固定目标
            elapsed = time.perf_counter() - start
            if elapsed < FIXED_TARGET_TIME:
                time.sleep(FIXED_TARGET_TIME - elapsed)
        else:
            raise ValueError("Unknown mode")

        # 网络抖动（独立于服务端实现的环境噪声）
        jitter = max(0.0, random.gauss(NETWORK_JITTER_MEAN, NETWORK_JITTER_STD))
        time.sleep(jitter)

        return time.perf_counter() - start


# ========== 攻击者工具函数 ==========
def cohen_d(x: List[float], y: List[float]) -> float:
    """Cohen's d 效应量（衡量均值差 / 合并标准差）"""
    mx, my = statistics.mean(x), statistics.mean(y)
    sx, sy = statistics.pstdev(x), statistics.pstdev(y)  # 总体标准差
    # 合并标准差（简化）
    s_pool = math.sqrt(((sx**2) + (sy**2)) / 2.0) if (sx > 0 or sy > 0) else 0.0
    return (mx - my) / s_pool if s_pool > 0 else float('inf') if mx != my else 0.0


def build_threshold(success_times: List[float], fail_times: List[float]) -> float:
    """
    基于画像样本，使用简单的两均值中点作为分类阈值。
    （在高斯近似 & 方差接近的场景下，这是接近 Bayes 最优的简单策略）
    """
    ms, mf = statistics.mean(success_times), statistics.mean(fail_times)
    return (ms + mf) / 2.0


def classify(t: float, threshold: float) -> bool:
    """返回 True 表示判定为成功，否则失败"""
    return t >= threshold  # 成功均值更大 → 高于阈值视为成功


def run_phase_profile(server: AuthServer, n_per_class: int) -> Tuple[List[float], List[float]]:
    """画像阶段：各采样 n 次，得到成功/失败的时间分布"""
    succ, fail = [], []
    for _ in tqdm(range(n_per_class), desc="Profiling (success)"):
        succ.append(server.one_request(True))
    for _ in tqdm(range(n_per_class), desc="Profiling (failure)"):
        fail.append(server.one_request(False))
    return succ, fail


def run_phase_infer(server: AuthServer, n: int, threshold: float) -> Tuple[float, float, float]:
    """
    推断阶段：对未知请求做分类，返回 (accuracy, mean_success, mean_fail)
    其中 mean_* 是在这批未知样本里按真实标签分桶后的均值（用于报告）
    """
    y_true, y_pred, times = [], [], []
    for _ in tqdm(range(n), desc="Inferring"):
        # 随机生成真实标签（真实世界里“成功”较少，这里用 p=0.3 可自行调节）
        real_success = (random.random() < 0.3)
        t = server.one_request(real_success)
        times.append(t)
        y_true.append(real_success)
        y_pred.append(classify(t, threshold))

    # 统计准确率
    correct = sum(int(a == b) for a, b in zip(y_true, y_pred))
    acc = correct / n

    # 分桶均值（用于报告观测差异）
    succ_times = [t for t, y in zip(times, y_true) if y]
    fail_times = [t for t, y in zip(times, y_true) if not y]
    ms = statistics.mean(succ_times) if succ_times else float('nan')
    mf = statistics.mean(fail_times) if fail_times else float('nan')
    return acc, ms, mf


def report_distribution(name: str, succ: List[float], fail: List[float]):
    ms, mf = statistics.mean(succ), statistics.mean(fail)
    ss, sf = statistics.pstdev(succ), statistics.pstdev(fail)
    d = cohen_d(succ, fail)
    print(f"\n[{name}] 画像分布统计")
    print(f"  成功: mean={ms*1000:.3f} ms, std={ss*1000:.3f} ms, n={len(succ)}")
    print(f"  失败: mean={mf*1000:.3f} ms, std={sf*1000:.3f} ms, n={len(fail)}")
    print(f"  均值差: {(ms - mf)*1000:.3f} ms, Cohen's d={d:.3f}")


def run_all(mode: str):
    print("\n" + "="*70)
    print(f"服务端模式: {mode}")
    server = AuthServer(mode=mode)

    # 画像阶段
    succ_prof, fail_prof = run_phase_profile(server, PROFILE_SAMPLES_PER_CLASS)
    report_distribution(f"{mode}/画像", succ_prof, fail_prof)

    # 阈值
    thr = build_threshold(succ_prof, fail_prof)
    print(f"  画像得到的分类阈值: {thr*1000:.3f} ms")

    # 推断阶段
    acc, ms, mf = run_phase_infer(server, INFER_SAMPLES, thr)
    print(f"\n[{mode}] 推断阶段结果（未知样本）")
    print(f"  推断准确率: {acc*100:.2f}%  （50%≈随机猜测）")
    print(f"  未知样本内观测: 成功均值={ms*1000:.3f} ms, 失败均值={mf*1000:.3f} ms, 均值差={(ms-mf)*1000:.3f} ms")


if __name__ == "__main__":
    # 对比：随机延迟 vs 固定延迟
    run_all("random_delay")
    run_all("fixed_delay")
