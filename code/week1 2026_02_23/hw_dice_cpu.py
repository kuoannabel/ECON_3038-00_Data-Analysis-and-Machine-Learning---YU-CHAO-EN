import psutil
import time
from tqdm import tqdm
#to know if still running : add cpu%
def monitor_cpu():
    print(f"CPU usage: {psutil.cpu_percent()}%")

def two_dice_sample_space_with_cpu_monitor(rolls):
    dice = [1,2,3,4,5,6]
    twodice = [[i,j] for i in dice for j in dice]

    ss = [[]]

    start_time = time.time()
    for i in range(rolls):
        new_ss = []
        for j in tqdm(range(len(ss)), desc=f"Rolling {i+1}"):
            new_ss.extend([ss[j] + [outcome] for outcome in twodice])

            # 每 10000 個結果印一次 CPU 使用率
            if j % 10000 == 0:
                print(f"[Roll {i+1}] CPU usage: {psutil.cpu_percent()}%")

        ss = new_ss
        print(f"After rolling {i+1} times: Total outcomes = {len(ss)}")
    
    total_time = time.time() - start_time
    print(f"Finished {rolls} rolls in {total_time:.2f} seconds")
    print("Expected total outcomes:", 36**rolls)


two_dice_sample_space_with_cpu_monitor(6)
