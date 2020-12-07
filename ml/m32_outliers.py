#2020-12-07
#이상치 처리
#무조건 맹신하지 말 것. 중요한 정보가 날아갈 수도 있다.
#판단 잘해야 함. 

import numpy as np

def outliers(data_out):
    #4분의 1지점과 4분의 3지점(전체 데이터 길이의)
    #그 둘을 뺀 값들을 *1.5로 데이터 범위 잡기 
    
    #25프로, 75프로 언패킹 -> 데이터 4등분 
    quartile_1, quartile_3 = np.percentile(data_out, [25, 75])
    print("1사분위: ", quartile_1) #3.25
    print("3사분위: ", quartile_3) #97.5

    iqr = quartile_3 - quartile_1 #94.25

    #이상치를 판단하는 가장 유명한 식
    lower_bound = quartile_1 - (iqr * 1.5) 
    upper_bound = quartile_3 + (iqr * 1.5) 
    #100퍼센트 범위를 좀 넘어가지만 그 정도까지 유도리를 주겠다 

    #1.5배수 넘어가는 지역 or 하단의 지점을 벗어나는 놈들을 찾아서 return
    return np.where((data_out > upper_bound) | (data_out < lower_bound))


a = np.array([1, 2, 3, 4, 10000, 6, 7, 5000, 90, 100],
            [1, 3, ...]) #10개 중에 두 개나 날리기... 판단 잘해야 함. 예제니까 그냥 하지만...
b = outliers(a) #전체 데이터:1~10000 중에 1사분위는 3.25고 97.5 이상부터는 이상한 놈이다 
                #따라서 1.5배수까지 유도리 있게 주겠다 

print("이상치의 위치: ", b) #이상치의 위치:  (array([4, 7], dtype=int64),) -> 10000과 5000이 이상치다


#숙제: column이 여러 개일 때는 함수를 어떻게 변형해서 사용할까? 