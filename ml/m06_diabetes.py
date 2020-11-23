#2020-11-23 (11일차)
#Machine Learning: diabetes

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler #robust = 이상치 제거에 효과적


#iris는 분류이므로 _classifier만 사용 (총 모델 4개)
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor 
                                                   #사이킷런의 이름을 보고 이해하는 능력 필요
                                                   #classifier: 분류 regressor: 회귀
                                                   #cf. logistic regressor는 regressor지만 분류

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor #feature importance
from sklearn.model_selection import train_test_split
                                    
#1. 데이터
x, y = load_diabetes(return_X_y=True)

feature_names = load_diabetes().feature_names # ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
print(feature_names)

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

# metrics_score = accuracy_score(y_test, y_predict)
# print("accuracy_score: " , metrics_score)

metrics_score = r2_score(y_test, y_predict)
print("r2_score: ", metrics_score)

print(y_test[:10], "의 예측 결과: \n", y_predict[:10])



# LinearSVC
# model.score:  0.0
# r2_score:  -0.05417705666534589
# [235. 150. 124.  78. 168. 253.  97. 102. 249. 142.] 의 예측 결과:
#  [ 91. 281. 281.  90.  90.  91.  53.  87. 230. 200.]


# SVC
# model.score:  0.0
# r2_score:  -0.10922943120678053
# [235. 150. 124.  78. 168. 253.  97. 102. 249. 142.] 의 예측 결과:
#  [ 91. 220.  91.  90.  90.  91.  53.  90. 220. 200.]

# KNeighborsClassifier
# model.score:  0.0
# r2_score:  -0.5574668030667056
# [235. 150. 124.  78. 168. 253.  97. 102. 249. 142.] 의 예측 결과:
#  [ 91. 129.  77.  64.  60.  63.  42.  53.  67.  74.]


# model = KNeighborsRegressor()
# model.score:  0.3968391279034368
# r2_score:  0.3968391279034368
# [235. 150. 124.  78. 168. 253.  97. 102. 249. 142.] 의 예측 결과:
#  [166.4 190.6 124.  132.6 120.4 120.2 101.4 108.6 145.4 131.6]


# model = RandomForestClassifier()
# model.score:  0.011235955056179775
# r2_score:  -0.0028884439482794733
# [235. 150. 124.  78. 168. 253.  97. 102. 249. 142.] 의 예측 결과:
#  [ 91. 220.  77.  64.  97. 138.  83. 214. 229.  70.]



# model = RandomForestRegressor() 
# model.score:  0.3789691221066297
# r2_score:  0.3789691221066297
# [235. 150. 124.  78. 168. 253.  97. 102. 249. 142.] 의 예측 결과:
#  [164.76 204.25 148.97 127.15 103.94 122.35  93.23 142.87 141.72 131.42]



