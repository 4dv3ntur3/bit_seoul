#과제
#이 함수 완벽하게 이해하기 


import numpy as np
dataset = np.array(range(1, 11))
size = 5

def split_x(seq, size):
    aaa = [] #는 테스트
    for i in range(len(seq)-size+1):
        subset = seq[i:(i+size)]
        aaa.append([item for item in subset])
        #aaa.append 줄일 수 있음
        #

    print(type(aaa))
    return np.array(aaa)

dataset = split_x(dataset, size)
print("=============")
print(dataset)


