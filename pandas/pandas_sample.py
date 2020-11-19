#2020-11-19 
#pandas
#loc, iloc******* 중요

import pandas as pd
import numpy as np

from numpy.random import randn
np.random.seed(100) #난수

data = randn(5, 4) #5행 4열
print(data)

#pandas -> dataframe, series
#dataframe(행렬과 비슷한 개념)
df = pd.DataFrame(data, index='A B C D E'.split(), #A B C D E = index
                        columns='가 나 다 라'.split()) #가 나 다 라 = columns (header)


print(df)

'''
          가         나         다         라
A -1.749765  0.342680  1.153036 -0.252436
B  0.981321  0.514219  0.221180 -1.070043
C -0.189496  0.255001 -0.458027  0.435163
D -0.583595  0.816847  0.672721 -0.104411
E -0.531280  1.029733 -0.438136 -1.118318
'''

#5행 4열
data2 = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], 
         [13, 14, 15, 16], [17, 18, 19, 20]] #list

df2 = pd.DataFrame(data2, index=['A', 'B', 'C', 'D', 'E'], 
                          columns=['가', '나', '다', '라'])


print(df2)
'''
    가   나   다   라
A   1   2   3   4
B   5   6   7   8
C   9  10  11  12
D  13  14  15  16
E  17  18  19  20
'''


df3 = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6]])) # 2, 3
#자동으로 index랑 header 넣어서 출력해 준다
print(df3) 

'''
   0  1  2
0  1  2  3
1  4  5  6
'''


#Column별로 잘라서 출력 가능
print("df2['나']: \n", df2['나']) #2, 6, 10, 14, 18
'''
df2['나']:
 A     2
B     6
C    10
D    14
E    18
Name: 나, dtype: int64
'''



print("d2f['나', '라']: \n", df2[['나', '라']]) #2 6 10 14 18
                                                #4 8 12 16 20
'''
d2f['나', '라']:
     나   라
A   2   4
B   6   8
C  10  12
D  14  16
E  18  20
'''


#실행하기 전에 생각해 보기
#loc, iloc******** + csv 읽어들이기

# print("df2[0]: ", df2[0]) #0이라는 column이 없다고 판단, 에러. column 이름을 넣어 줘야.
# print("df2.loc['나']: \n", df2.loc['나']) #에러. location에는 column명 들어가면 안 됨. loc -> 행, 열 순으로 나감. 즉, 위치로 찾음

print("df.iloc[:, 2]: \n", df2.iloc[:, 2]) #모든 행의 세 번째 index 0, 1, 2
'''
df.iloc[:, 2]:
 A     3
B     7
C    11
D    15
E    19
Name: 다, dtype: int64
'''

# print("df[:, 2]: \n", df2[:, 2]) #모든 행의 세 번째 index 0, 1, 2
#error: iloc는 위치에 맞춘 순서대로 
#numpy에선 가능한데 pandas에선 안 된다 


#ROW(열)
print("df2.loc['A']\n", df2.loc['A'])
'''
df2.loc['A']
 가    1
나    2
다    3
라    4
Name: A, dtype: int64
'''

print("df2.loc[['A, 'C']]\n", df2.loc[['A', 'C']])
'''
df2.loc['A, 'C']
    가   나   다   라
A  1   2   3   4
C  9  10  11  12
'''



print("df2.iloc[0]\n", df2.iloc[0])
print("df2.iloc[[0, 2]]\n", df2.iloc[[0, 2]])
'''
df2.iloc[0]
 가    1
나    2
다    3
라    4
Name: A, dtype: int64
df2.iloc[[0, 2]]
    가   나   다   라
A  1   2   3   4
C  9  10  11  12
'''



#행렬
#loc -> index or header
print("df2.loc[['A', 'B'], ['나', '다']]:\n", df2.loc[['A', 'B'], ['나', '다']])
'''
df2.loc[['A', 'B'], ['나', '다']]:
    나  다
A  2  3
B  6  7
'''


#하이라이트
#1개의 값 확인
print("df2.loc['E', '다']:\n", df2.loc['E', '다']) 
'''
df2.loc['E', '다']:
 19
'''

print("df2.iloc[4, 2]:\n", df2.iloc[4, 2]) #index는 0부터 시작 = 5행 3열
'''
df2.iloc[4, 2]:
 19
'''

print("df2.iloc[4][2]:\n", df2.iloc[4][2]) #index는 0부터 시작 = 즉 5행 3열
'''
df2.iloc[4][2]:
 19
'''