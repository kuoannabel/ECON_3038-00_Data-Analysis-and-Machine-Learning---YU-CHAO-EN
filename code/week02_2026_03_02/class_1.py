def have_six():
    one = []
    for i in range(1,7):
       
            one.append(i)#rolling two dice
    #print(one)

    ss = []
    for a in range(len(one)):
        for b in range(len(one)):
            for c in range(len(one)):
                for d in range(len(one)):
                    
                    ss.append([one[a], one[b], one[c], one[d]])#rolling two dice twice[first time, second time]
    """            
    for i in range(len(ss)):
        print(ss[i],end = "\n")
    """
    print("len(ss):", len(ss))
    cnt = 0
    print("pow(6,4):", pow(6,4))
    for i in range(len(ss)):
        #print("6 in ss[{}]:".format(i), 6 in ss[i])
        if 6 in ss[i]:
            cnt += 1
    print("6 cnt:", cnt)
    #first six
    """
    print("#first six ss[5]:", ss[5])
    print("6 in ss[5]:", 6 in ss[5])
    print("6 in ss[6]:", 6 in ss[6])
    """
    P = cnt / len(ss)
    print("\n[precise] P(6 in ss):\n",P)#非逼近值
    print(1 - pow(5,4) / pow(6,4))
    print("\nDe Mere: “At least one 6 in 4 rolls” vs “at least one double-six in 24 throws of 2 dice” ") 
    print("in one dice:", 1 - pow(5,4) / pow(6,4))#1 - no six in 4 rolls
    print("in two dice:", 1 - pow(35,24) / pow(36,24))#1 - no double six in 24 throws of 2 dice
    
have_six()