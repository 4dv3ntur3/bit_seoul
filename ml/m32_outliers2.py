#과제: column이 여러 개일 때는 함수를 어떻게 변형해서 사용할까? 
#np.percentile: 백분위 수 구하기
#백분위수(Percentile): 오름차순 정렬했을 때 0을 최소값, 100을 최대값으로 백분율로 나타낸 특정 위치 값이다.
#사분위수는 25, 50, 75를 기준 점으로 나눠져 1분위부터 4분위까지 존재하게 된다. 
#백분위수 q 값은 0부터 100 값을 사용. (0.0~1.0가 아니므로 주의)


#탐지만. 삭제할지 말지는 우리가 판단.
#이상치 처리에는 제거만 있는 게 아니다. 보간도 있고 여러 가지 있음. 

import numpy as np

def outliers(data_out):
    #4분의 1지점과 4분의 3지점(전체 데이터 길이의)
    #그 둘을 뺀 값들을 *1.5로 데이터 범위 잡기 
    
    #25프로, 75프로 언패킹 -> 데이터 4등분 
                                                    #이 부분만 좀 수정해 주면 됨 이 부분은 절대적인 게 아니다 
    row = data_out.shape[0] #행
    col = data_out.shape[1] #열

    where = []
    
    #각 열별로 data 훑으면서 범위 나간 것 찾는다
    for v in range(col):
        data = data_out[:, v]
        quartile_1, quartile_3 = np.percentile(data, [25, 75])
        
        print("1사분위: ", quartile_1) #3.25
        print("3사분위: ", quartile_3) #97.5

        iqr = quartile_3 - quartile_1 #94.25

        #이상치를 판단하는 가장 유명한 식
        lower_bound = quartile_1 - (iqr * 1.5) 
        upper_bound = quartile_3 + (iqr * 1.5) 
    
        where.append(np.where((data > upper_bound) | (data < lower_bound)))

    #100퍼센트 범위를 좀 넘어가지만 그 정도까지 유도리를 주겠다 
    #1.5배수 넘어가는 지역 or 하단의 지점을 벗어나는 놈들을 찾아서 그 위치를 return

    return where


a = np.array([[1, 2, 3, 4, 10000, 6, 7, 5000, 90, 100],
              [10000, 20000, 3, 40000, 50000, 60000, 70000, 8, 90000, 200000]]) #이런 식으로 행렬 데이터 넣었을 때도 사용 가능하게 변형 

a = a.T 

print(a)
print(a.shape)
            
            
                #10개 중에 두 개나 날리기... 판단 잘해야 함. 예제니까 그냥 하지만...
b = outliers(a) #전체 데이터:1~10000 중에 1사분위는 3.25고 97.5 이상부터는 이상한 놈이다 
                #따라서 1.5배수까지 유도리 있게 주겠다 

print("이상치의 위치: ", b) #이상치의 위치:  (array([4, 7], dtype=int64),) -> 10000과 5000이 이상치다

#이상치 제거는?