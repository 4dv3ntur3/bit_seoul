#2020-11-23 (11일차)
#Machine Learning: boston

import numpy as np
from sklearn.datasets import load_boston
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
x, y = load_boston(return_X_y=True)

feature_names = load_boston().feature_names #['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT']
print(feature_names)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=66, shuffle=True, train_size=0.8 
)

scaler = StandardScaler()
scaler.fit(x_train)

scaler.transform(x_train)
scaler.transform(x_test)



#2. 모델(일단은 dafault만 미세 조정은 추후에)
model = LinearSVC()
# model = SVC()
# model = KNeighborsClassifier()
# model = KNeighborsRegressor()
# model = RandomForestClassifier()
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

# metrics_score = r2_score(y_test, y_predict)
# print("r2_score: ", metrics_score)

print(y_test[:10], "의 예측 결과: \n", y_predict[:10])






# LinearSVC
# model.fit(x_train, y_train)
#   File "C:\Anaconda3\lib\site-packages\sklearn\neighbors\_base.py", line 1146, in fit
#     check_classification_targets(y)
#   File "C:\Anaconda3\lib\site-packages\sklearn\utils\multiclass.py", line 172, in check_classification_targets
#     raise ValueError("Unknown label type: %r" % y_type)
# ValueError: Unknown label type: 'continuous'
# -> 분류를 위해서는 data type이 int여야 하는데 float?라서 나는 오류

# SVC
# model.fit(x_train, y_train)
#   File "C:\Anaconda3\lib\site-packages\sklearn\neighbors\_base.py", line 1146, in fit
#     check_classification_targets(y)
#   File "C:\Anaconda3\lib\site-packages\sklearn\utils\multiclass.py", line 172, in check_classification_targets
#     raise ValueError("Unknown label type: %r" % y_type)
# ValueError: Unknown label type: 'continuous'


# KNeighborsClassifier
# model.fit(x_train, y_train)
#   File "C:\Anaconda3\lib\site-packages\sklearn\neighbors\_base.py", line 1146, in fit
#     check_classification_targets(y)
#   File "C:\Anaconda3\lib\site-packages\sklearn\utils\multiclass.py", line 172, in check_classification_targets
#     raise ValueError("Unknown label type: %r" % y_type)
# ValueError: Unknown label type: 'continuous'


# model = KNeighborsRegressor()
# model.score:  0.5900872726222293
# r2_score:  0.5900872726222293
# [16.3 43.8 24.  50.  20.5 19.9 17.4 21.8 41.7 13.1] 의 예측 결과:
#  [17.32 25.16 21.56 38.58 20.48 23.58 21.9  21.14 35.44 19.36]


# model = RandomForestClassifier()
# model.fit(x_train, y_train)
#   File "C:\Anaconda3\lib\site-packages\sklearn\neighbors\_base.py", line 1146, in fit
#     check_classification_targets(y)
#   File "C:\Anaconda3\lib\site-packages\sklearn\utils\multiclass.py", line 172, in check_classification_targets
#     raise ValueError("Unknown label type: %r" % y_type)
# ValueError: Unknown label type: 'continuous'



# model = RandomForestRegressor() 
# model.score:  0.9208771987636387
# r2_score:  0.9208771987636387
# [16.3 43.8 24.  50.  20.5 19.9 17.4 21.8 41.7 13.1] 의 예측 결과:
#  [14.503 46.64  28.555 45.636 21.236 21.085 19.566 20.475 45.193 16.549]
 