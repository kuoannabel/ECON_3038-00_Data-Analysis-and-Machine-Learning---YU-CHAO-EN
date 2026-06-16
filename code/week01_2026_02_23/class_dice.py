def dice1_4t_sample_space():
    ss = []
    print(type(ss))
    #one dice rolls four times
    for i in range(1,7):
        for j in range(1, 7):
            for k in range(1, 7):
                for l in range(1, 7):
                    ss.append([i, j, k, l])
    print(ss)
    print(len(ss))
    print(pow(6,4))
  
    
dice1_4t_sample_space()

