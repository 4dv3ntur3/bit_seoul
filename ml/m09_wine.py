#2020-11-23 (11일차)
#Machine Learning: load_wine

import numpy as np
from sklearn.datasets import load_wine
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
# x, y = load_wine(return_X_y=True)
datasets = load_wine()
x = datasets.data
y = datasets.target

print(datasets.feature_names)
#['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']

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

metrics_score = accuracy_score(y_test, y_predict)
print("accuracy_score: " , metrics_score)

# metrics_score = r2_score(y_test, y_predict)
# print("r2_score: ", metrics_score)

print(y_test[:10], "의 예측 결과: \n", y_predict[:10])



# LinearSVC
# model.score:  0.9444444444444444
# accuracy_score:  0.9444444444444444
# [2 1 1 0 1 1 2 0 0 1] 의 예측 결과:
#  [2 1 1 0 1 1 2 0 0 1]


# SVC
# model.score:  0.6944444444444444
# accuracy_score:  0.6944444444444444
# [2 1 1 0 1 1 2 0 0 1] 의 예측 결과:
#  [1 2 1 0 1 1 2 0 0 2]

# KNeighborsClassifier
# model.score:  0.6944444444444444
# accuracy_score:  0.6944444444444444
# [2 1 1 0 1 1 2 0 0 1] 의 예측 결과:
#  [1 2 1 0 1 1 2 0 0 2]

# model = KNeighborsRegressor()
# ValueError: Classification metrics can't handle a mix of multiclass and continuous targets

# model = RandomForestClassifier()
# model.score:  1.0
# accuracy_score:  1.0
# [2 1 1 0 1 1 2 0 0 1] 의 예측 결과:
#  [2 1 1 0 1 1 2 0 0 1]


# model = RandomForestRegressor() 
# ValueError: Classification metrics can't handle a mix of multiclass and continuous targets


