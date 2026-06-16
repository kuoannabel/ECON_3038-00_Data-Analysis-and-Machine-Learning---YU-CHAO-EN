def two_dice_sample_space(rolls):
    dice = [1,2,3,4,5,6]

    # 單次兩顆骰子
    twodice = []
    for i in dice:
        for j in dice:
            twodice.append([i,j])#[1,1]

    ss = [[]]   # 起點：什麼都還沒 roll

    for i in range(rolls):
        new_ss = []#[]
        for j in range(len(ss)):
            for outcome in twodice:#每丟一次 ss中的每一個結果，都要配上 outcome 的全部可能 因為那是那一次的所有可能結果
                new_ss.append(ss[j] + [outcome])#避免影響到len(ss) 把ss的結果加上outcome 產生新的結果放到new_ss中
                #print(ss[j] + [outcome])
        ss = new_ss#再放回ss中，準備下一輪的迭代
        print("After rolling {} times:".format(i+1))
        #print(ss)

    #print(ss)
    print("Total outcomes:", len(ss))
    print("Expected:", 36**rolls)


def how_many_rolls():
    for i in range(1, 7):
        two_dice_sample_space(i)
#answer:can do only 5 times 
how_many_rolls()
  
