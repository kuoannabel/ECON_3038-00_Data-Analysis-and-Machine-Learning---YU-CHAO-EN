import numpy as np

# 1. 定義狀態空間與目標分佈 (pi(x) 比例)
states = np.array([1, 2, 3, 4, 5, 6, 7])
target = lambda x: x  # 我們的目標是機率與數值成正比

# 2. 建立轉移矩陣 P
n_states = len(states)
P = np.zeros((n_states, n_states))

for i, x in enumerate(states):
    # 潛在的建議移動 (左或右)
    if x == 1:
        proposals = [1, 2]
    elif x == 7:
        proposals = [6, 7]
    else:
        proposals = [x-1, x+1]
    
    prob_proposal = 1.0 / len(proposals)
    
    for proposed in proposals:
        # 計算 Metropolis 接受率
        # Acceptance a = min(1, P(proposed)/P(current))
        acceptance = min(1, target(proposed) / target(x))
        
        j = proposed - 1  # 陣列索引 (0-6 對應 狀態1-7)
        
        if i == j:
            # 建議留在原地的機率 (邊界情況)
            P[i, i] += prob_proposal
        else:
            # 成功移動的機率
            P[i, j] += prob_proposal * acceptance
            # 被拒絕而留在原地的機率
            P[i, i] += prob_proposal * (1 - acceptance)

# 3. 設定初始分佈 (t=0, 狀態在 4)
pi_0 = np.array([0, 0, 0, 1, 0, 0, 0])

# 4. 計算 t=3 的分佈: pi_3 = pi_0 * P^3
pi_3 = pi_0 @ np.linalg.matrix_power(P, 3)

print(f"在 t=3 時，各狀態的機率為：")
for s, p in zip(states, pi_3):
    print(f"P(theta={s}) = {p:.4f}")
    
