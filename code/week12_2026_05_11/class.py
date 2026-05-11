import numpy as np
import random
from matplotlib import pyplot as plt

pop = [1, 2, 3, 4, 5, 6, 7]
x = 4
n = 100  
random.seed(234)

# 用於紀錄每個點出現次數的字典
encounter_counts = {i: 0 for i in pop}
trajectory = [x]
encounter_counts[x] += 1

for i in range(n):
    # 1. 提議新狀態 (Proposal)
    if 1 < x < 7:
        proposed = random.choice([x - 1, x + 1])
    elif x == 1:
        proposed = random.choice([x, x + 1])
    elif x == 7:
        proposed = random.choice([x - 1, x])
    
    # 2. 計算接受率 (Acceptance Ratio)
    # 這裡目標分佈 P(x) 正比於 x
    acceptance_ratio = proposed / x
    
    # 3. 判定是否移動
    if acceptance_ratio >= 1 or random.random() < acceptance_ratio:
        x = proposed 
    
    trajectory.append(x)
    encounter_counts[x] += 1

# --- 輸出結果 ---
print("每個點被造訪的次數 (Encounter counts):")
for state, count in encounter_counts.items():
    print(f"狀態 {state}: {count} 次")

# --- 繪圖 ---
plt.figure(figsize=(12, 5))

# 子圖 1: 軌跡圖
plt.subplot(1, 2, 1)
plt.plot(trajectory, range(len(trajectory)), alpha=0.6)
plt.xlabel('State')
plt.ylabel('Time step')
plt.title('Trajectory of the Markov Chain')

# 子圖 2: 出現次數長條圖
plt.subplot(1, 2, 2)
plt.bar(encounter_counts.keys(), encounter_counts.values(), color='skyblue')
plt.xlabel('State')
plt.ylabel('Encounter Count')
plt.title('Distribution of States')

plt.tight_layout()
plt.show()