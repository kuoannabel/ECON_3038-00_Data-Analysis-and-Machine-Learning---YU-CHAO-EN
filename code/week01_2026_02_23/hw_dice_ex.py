def dice2_2t_sample_space():
    twodice = []
    for i in range(1,7):
        for j in range(1, 7):
            twodice.append([i, j])#rolling two dice
    print(twodice)

    ss = []
    for a in range(len(twodice)):
        for b in range(len(twodice)):
            ss.append([twodice[a] , twodice[b]])#rolling two dice twice[first time, second time]
    print(ss)
    print(len(ss))
    print(pow(36,2))
    
dice2_2t_sample_space(2)