#2020-11-23 (11일차)
#feature importance

import pandas as pd
wine = pd.read_csv('./data/csv/winequality-white.csv', sep=';', header=0)
y = wine['quality']
x = wine.drop('quality', axis=1) #quality를 뺀 나머지가 x

print(x.shape) #4898, 11
print(y.shape) #4898,


import numpy as np

#데이터 전처리의 일환
newlist = []
for i in list(y):
    if i<=4:
        newlist +=[0] # 4보다 작으면 labeling = 0

    elif i<=7:
        newlist +=[1] 
    
    else:
        newlist +=[2]


# 3부터 9까지의 데이터 분포, 5, 6, 7이 제일 많았다
# 3, 4, 5, 6, 7, 8, 9로 갈라서 본다 -> 0 1 2로 됨
# data 조작 아닙니까? 조작일 수도 있지만 전처리일 수도 있다
# wine의 품질 판단 데이터셋. wine의 등급이 3~9까지니까 맞추는 거였는데, 조절해서 0~2의 3단계로 줄임.

y = newlist

#2. 모델 만든 거 잇기
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


#1. 데이터

# data_np = np.loadtxt('./data/csv/winequality-white.csv', delimiter=';', skiprows=1) #head 제외하고 읽음


x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=66, shuffle=True, train_size=0.8
)

scaler = StandardScaler()
scaler.fit(x_train)

scaler.transform(x_train)
scaler.transform(x_test)



#2. 모델(일단은 dafault만 미세 조정은 추후에)
# model = LinearSVC()
# model = SVC()
# model = KNeighborsClassifier()
# model = KNeighborsRegressor()
model = RandomForestClassifier()
# model = RandomForestRegressor()


#3. 훈련
model.fit(x_train, y_train)


#4. 평가, 예측
from sklearn.metrics import r2_score, accuracy_score

score = model.score(x_test, y_test)     #회귀에서는 r2_score = score이다
                                        #accuracy_score != score (둘이 같다는 건 분류에서나 통하는 이야기)
print("model.score: ", score)



#accuracy_score를 넣어서 비교할 것
#회귀모델일 경우 r2_score와 비교할 것

y_predict = model.predict(x_test)

metrics_score = accuracy_score(y_test, y_predict)
print("accuracy_score: " , metrics_score)

# metrics_score = r2_score(y_test, y_predict)
# print("r2_score: ", metrics_score)


print(y_test[:10], "의 예측 결과: \n", y_predict[:10])


'''
model.score:  0.9489795918367347
accuracy_score:  0.9489795918367347
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1] 의 예측 결과:
 [1 1 1 1 1 1 1 1 1 1]


5, 6, 7을 1로 하니까 결과에 1이 많이 나옴
이런 식으로 data 뭉쳐서 accuracy 조정할 수 있다
data의 column을 맹목적으로 믿고 의지하면 안 됨! 
'''

