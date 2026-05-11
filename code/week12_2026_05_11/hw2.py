import numpy as np

#How much is P(theta=4) when t=3? 0.1501
#1. 定義狀態空間與目標分佈 (pi(x) 比例)
#states = np.array([1, 2, 3, 4, 5, 6, 7])
#target = lambda x: x  # 我們的目標是機率與數值成正比
#2. 建立轉移矩陣 P
#n_states = len(states)
#P = np.zeros((n_states, n_states))//for i, x in enumerate(states):
    # 潛在的建議移動 (左或右)
#    if x == 1:
#        proposals = [1, 2]
#    elif x == 7:
#        proposals = [6, 7]
#    else:
#        proposals = [x-1, x+1]
#    prob_proposal = 1.0 / len(proposals)
#    for proposed in proposals:
#        # 計算 Metropolis 接受率
#        # Acceptance a = min(1, P(proposed)/P(current))
#        acceptance = min(1, target(proposed) / target(x))
#        j = proposed - 1  # 陣列索引 (0-6 對應 狀態1-7)
#        if i == j:
#            # 建議留在原地的機率 (邊界情況)
#            P[i, i] += prob_proposal
#        else:
#            # 成功移動的機率
#            P[i, j] += prob_proposal * acceptance
#            # 被拒絕而留在原地的機率
#            P[i, i] += prob_proposal * (1 - acceptance)
#3. 設定初始分佈 (t=0, 狀態在 4)
#pi_0 = np.array([0, 0, 0, 1, 0, 0, 0])
#4. 計算 t=3 的分佈: pi_3 = pi_0 * P^3
#pi_3 = pi_0 @ np.linalg.matrix_power(P, 3)
#print(f"在 t=3 時，各狀態的機率為：")
#for s, p in zip(states, pi_3):
#    print(f"P(theta={s}) = {p:.4f}")


def calculate_mcmc_exact_prob(target_steps=3):
    # 1. 定義狀態空間 (1 到 7)
    states = np.array([1, 2, 3, 4, 5, 6, 7])
    n = len(states)
    
    # 2. 建立轉移矩陣 T (7x7)
    # T[i, j] 代表從狀態 i+1 轉移到狀態 j+1 的機率
    T = np.zeros((n, n))
    
    for i in range(n):
        curr_x = states[i]
        
        # 決定建議分佈 (Proposal distribution)
        if curr_x == 1:
            proposals = [1, 2]
        elif curr_x == 7:
            proposals = [6, 7]
        else:
            proposals = [curr_x - 1, curr_x + 1]
        
        prob_proposal = 1.0 / len(proposals)
        
        for proposed_x in proposals:
            j = proposed_x - 1  # 目標狀態的索引
            
            # 計算 Metropolis 接受率: min(1, P(proposed)/P(current))
            # 這裡的目標分佈 P(x) 就是 x 本身
            acceptance_rate = min(1.0, proposed_x / curr_x)
            
            if i == j:
                # 情況 A: 建議留在原地 (邊界情況)
                T[i, i] += prob_proposal
            else:
                # 情況 B: 建議移動到另一個狀態
                move_prob = prob_proposal * acceptance_rate
                stay_prob = prob_proposal * (1 - acceptance_rate)
                
                T[i, j] += move_prob  # 成功移動
                T[i, i] += stay_prob  # 被拒絕，留在原地

    # 3. 設定初始機率向量 w (t=0 時，100% 在狀態 4)
    # 索引 3 對應狀態 4
    w = np.zeros(n)
    w[3] = 1.0
    
    print(f"初始狀態 w(t=0): {w}")
    print("-" * 30)

    # 4. 利用矩陣乘法計算每一代的機率分佈
    # w_next = w_current @ T
    current_w = w
    for t in range(1, target_steps + 1):
        current_w = np.dot(current_w, T)
        print(f"時間步長 t={t} 的機率分佈:")
        for s, prob in zip(states, current_w):
            print(f"  P(theta={s}) = {prob:.4f}")
        print("-" * 30)

    return current_w

# 執行計算
final_probs = calculate_mcmc_exact_prob(3)

print(f"\n最終結論: 在 t=3 時 P(theta=4) 的精確機率為: {final_probs[3]:.4f}")