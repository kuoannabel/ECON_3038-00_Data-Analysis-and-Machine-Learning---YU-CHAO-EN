import numpy as np
from matplotlib import pyplot as plt
import math

# 正負數通用的無條件捨去兩位函數
def truncate2(x):
    if x >= 0:
        return math.floor(x * 100) / 100  # 正數用 floor
    else:
        return math.ceil(x * 100) / 100   # 負數用 ceil
    
bin_cnt = 100
np.random.seed(234)
data = np.random.normal(0, 1, size=100000)

# 計算95% HDI (用quantile近似)
left = np.quantile(data, 0.025)
right = np.quantile(data, 0.975)



est_left_trunc = truncate2(left)
est_right_trunc = truncate2(right)
print("\n### Estimated HDI ###")
print("est Left limit:", est_left_trunc)
print("est Right limit:", est_right_trunc)
print("\n")

# histogram
counts, bins = np.histogram(data, bins=bin_cnt)
for i in range(len(counts)):
    print(f"Bin {i+1}: {bins[i]} ~ {bins[i+1]} --> Count = {counts[i]}")
cnt = 0

protion = counts / np.sum(counts)
left_sum = 0
for i in range(len(protion)):
    left_sum += protion[i]
    if left_sum >= 0.025:
        left_limit_idx = i 
        left_limit = bins[i]
        break

right_sum = 0
for i in range(len(protion)-1, -1, -1):
    right_sum += protion[i]
    if right_sum >= 0.025:
        right_bin = i
        right_limit = bins[i]
        break

# 對 histogram HDI 也無條件捨去兩位
left_limit_trunc = truncate2(left_limit)
right_limit_trunc = truncate2(right_limit)

print("\n### HDI limits from histogram ###")
print(f"HDI left limit (histogram): {left_limit} --> truncate: {left_limit_trunc} -->bin: left_bin = {left_limit_idx+1}")
print(f"HDI right limit (histogram): {right_limit} --> truncate: {right_limit_trunc} -->bin: right_bin = {right_bin}")


print("\n")
for i in range(left_limit_idx+1, right_bin + 1):#range(left_bin, right_bin + 1) 是因為 []是左閉右開區間，所以要加1才能包含 right_bin 的 bin]
    #print(f"Bin {i} is in HDI region: {bins[i]} ~ {bins[i+1]} --> Count = {counts[i]}")# 計算 HDI 區間的 count
    cnt += counts[i]
print(f"\nCount in HDI region: {cnt}")
 




print(f"\nCount in left_bin:{left_limit_idx+1} left_limit: {bins[left_limit_idx]} ~ right_bin:{right_bin} right_limit: {bins[right_bin]} region: {cnt}")

cnt_test = 0
test_left_bin = 24
test_left_limit_idx = test_left_bin - 1 # 因為 bins[i-1] ~ bins[i] 是第 i 個 bin，所以要減1才能得到對應的 bin index
test_right_bin = 68

print(f"\n### TEST ### \n left bin :{test_left_bin}   right bin: {test_right_bin} ")
for i in range(test_left_bin, test_right_bin + 1):#range(25, 69) 是因為 bins[i] ~ bins[i+1] 是第 i+1 個 bin
    cnt_test += counts[i]
print(f"Count in {test_left_bin}:{bins[test_left_limit_idx]} ~ {test_right_bin}: {bins[test_right_bin]} region: {cnt_test}")






def draw_histogram(data):
    plt.hist(data, bins=bin_cnt)
    # 標註每個 bin 的 count
    for i in range(len(counts)):
        plt.text(bins[i] + (bins[1]-bins[0])/2, counts[i]+500, str(counts[i]),
                 ha='center', va='bottom', fontsize=9)

    # 標出 HDI 左右端線 (truncate)
    plt.axvline(left_limit_trunc, color='red', linestyle='--', linewidth=2, label=f'HDI left ({left_limit_trunc})')
    plt.axvline(right_limit_trunc, color='red', linestyle='--', linewidth=2, label=f'HDI right ({right_limit_trunc})')

    # 標出 HDI 區間陰影
    plt.fill_betweenx([0, max(counts)], left_limit_trunc, right_limit_trunc, color='red', alpha=0.2)

    # 標出 mean
    mean_val = np.mean(data)
    plt.axvline(mean_val, color='green', linestyle='-', linewidth=2, label=f'Mean ({mean_val})')

    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Histogram with 95% HDI")
    plt.ylim(0, max(counts)*1.1)
    plt.legend()
    plt.show()

draw_histogram(data)
