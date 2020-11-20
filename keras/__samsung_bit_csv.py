######## *.csv
#데이터

import numpy as np
import pandas as pd

#csv 불러오기
#한글 깨짐: encoding='CP949'
samsung = pd.read_csv('./data/csv/삼성전자 1120.csv', encoding='CP949')
# print(samsung) #<class 'pandas.core.frame.DataFrame'>

bit = pd.read_csv('./data/csv/비트컴퓨터 1120.csv', encoding='CP949')
# print(bit) #<class 'pandas.core.frame.DataFrame'>

gold = pd.read_csv('./data/csv/금현물.csv', encoding='CP949')
# kos = pd.read_csv('./data/csv/코스닥.csv', encoding='CP949')




#날짜별로 오름차순 정렬
samsung = samsung.sort_values(['일자'], ascending=['True'])
bit = bit.sort_values(['일자'], ascending=['True'])
gold = gold.sort_values(['일자'], ascending=['True'])
# kos = kos.sort_values(['일자'], ascending=['True'])


samsung = samsung[["시가", "고가", "저가", "개인", "종가"]]
bit = bit[["시가", "고가", "종가"]]
gold = gold[["시가", "고가", "저가", "종가"]]
# kos = kos[["시가", "고가", "저가", "현재가"]]



for i in range(len(samsung.index)):
    for j in range(len(samsung.iloc[i])):
        samsung.iloc[i, j] = int(samsung.iloc[i, j].replace(',', ''))


for i in range(len(bit.index)):
    for j in range(len(bit.iloc[i])):
        bit.iloc[i, j] = int(bit.iloc[i, j].replace(',', ''))

for i in range(len(gold.index)):
    for j in range(len(gold.iloc[i])):
        gold.iloc[i, j] = int(gold.iloc[i, j].replace(',', ''))


# for i in range(len(kos.index)):
#     for j in range(len(kos.iloc[i])):
#         kos.iloc[i, j] = int(kos.iloc[i, j].replace(',', ''))


# 데이터 정리
# 2020/11/20 out
# 2018/03/19~2018/05/03 out


samsung = samsung[(samsung["시가"] < 2000000)]
samsung = samsung[:len(samsung.index)-1]

bit = bit[:len(bit.index)-1]

#*.npy 저장
np.save('./data/samsung.npy', arr=samsung)
np.save('./data/bit.npy', arr=bit)
np.save('./data/gold.npy', arr=gold)
# np.save('./data/kos.npy', arr=kos)

