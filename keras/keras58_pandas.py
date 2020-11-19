#2020-11-19 (9일차)
#pandas
# *.csv

import numpy as np
import pandas as pd

#csv를 pandas 형식으로 읽어들여 datasets에 넣겠다
#header와 index 어떻게 잡을지
datasets = pd.read_csv('./data/csv/iris_ys.csv', 
                        header=0, index_col=0, sep=',') #첫 번째 행이 header다, index_col=0 (1, 2, 3, 4... 맨 왼쪽에 index column 있었음), 콤마(,)로 구분되어 있었음
                        #header=None, index_col=None -> 151*6이 되면서 헤더와 인덱스도 실 데이터로 취급하게 됨
                        #sep 같은 경우는 csv default=','지만 가끔 ';'인 경우도 있으니 주의
                        
                        
print(datasets) 
#head, index는 실 데이터가 아니라고 판단, 사용자 편의상으로 출력해 주는 것일뿐 실 데이터로 가져오지 않음
print(datasets.shape) # 150, 5                                   150 rows * 5 columns (.shape()가 먹힌다)





#index_col= None, 0, 1 / header = None, 0, 1 일 때 총 9가지의 shape 경우의 수 정리

'''
header: 행 개수에 영향
index: 열 개수에 영향

header \ index   None               0                 1
===============================================================

 None            151, 6          151, 5             151, 5




 0               150, 6          150, 5             150, 5




 1               149, 6          149, 5             149, 5

'''


print(datasets.head()) #위에서 5개
print(datasets.tail()) #끝에서 5개
print(type(datasets)) #<class 'pandas.core.frame.DataFrame'>

#pandas가 numpy보다 조금 느리다 
#여태 정제된 data만 받음(사진도 수치로 된 데이터로 받고...)
#header나 index_col 잘못 넣어서 이상해지면? "맑음", "흐림" 이딴 식이면 수치도 아니고 단위가 섞여 있다거나... (섭씨 12도) 이런 식... 
#하지만 numpy의 경우는 중간에 숫자가 아닌 값이 있으면 안 됨 좀 더 정확히 말하자면 데이터 전체가 하나의 data type으로 통일되어 있어야 한다
#pandas의 경우는 숫자가 아닌 값(다른 값)이 있어도 불러올 수 있다




#pandas -> numpy
#to_numpy(): dtype parameter-dtype 설정 가능
#.values


aaa = datasets.to_numpy()
print(aaa)
print(type(aaa)) #<class 'numpy.ndarray'>
print(aaa.shape) #(150, 5)


np.save('./data/iris_ys_pd.npy', arr=aaa)

#pandas 공부할 것
