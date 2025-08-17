import time
import random
import statistics
from tqdm import tqdm

# 模拟认证接口
def auth_request(success: bool, mode: str):
    start = time.perf_counter()
    
    # 模拟内部逻辑不同的执行耗时
    if success:
        time.sleep(0.050)  # 成功路径额外 50ms
    else:
        time.sleep(0.020)  # 失败路径额外 20ms
    
    if mode == "random_delay":
        # 增加随机延迟 (100~300 ms)
        time.sleep(random.uniform(0.100, 0.300))
    elif mode == "fixed_delay":
        # 计算已用时间，补足到 300ms
        elapsed = time.perf_counter() - start
        target_time = 0.300
        if elapsed < target_time:
            time.sleep(target_time - elapsed)
    else:
        raise ValueError("Unknown mode")
    
    return time.perf_counter() - start


# 攻击者采样函数
def timing_attack_test(mode: str, samples: int = 200):
    success_times = []
    fail_times = []
    
    print(f"\n[模式: {mode}]")
    for _ in tqdm(range(samples), desc=f"采样成功路径"):
        success_times.append(auth_request(True, mode))
    for _ in tqdm(range(samples), desc=f"采样失败路径"):
        fail_times.append(auth_request(False, mode))
    
    mean_success = statistics.mean(success_times)
    mean_fail = statistics.mean(fail_times)
    
    print(f"成功路径 平均耗时: {mean_success*1000:.3f} ms")
    print(f"失败路径 平均耗时: {mean_fail*1000:.3f} ms")
    print(f"均值差: {(mean_success - mean_fail)*1000:.3f} ms")


if __name__ == "__main__":
    # 先看随机延迟的情况
    timing_attack_test("random_delay", samples=200)
    
    # 再看固定延迟的情况
    timing_attack_test("fixed_delay", samples=200)
