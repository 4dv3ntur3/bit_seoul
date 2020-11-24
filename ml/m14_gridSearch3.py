#2020-11-24 (12일차)
#gridsearch
#diabetes + randomforest



#수치 조정해 보기 & 각 parameter 찾아보기
#dictionary형태로 집어넣되 value는 list로 
#이 형태만 맞춰 주면 된다

# parameters = [
#     {'n_estimators': [30, 50, 70, 100, 150]},
#     {'max_depth':[2, 4, 6, 8, 10, 12]},
#     {'min_samples_leaf': [3, 5, 7, 10, 13, 15]},
#     {'min_samples_split': [2, 3, 5, 10, 11, 12]},
#     {'n_jobs': [-1]}
# ]

#위에는 그냥 5+6+6+6+1 
#얘 같은 경우는 2* 6* 4* 4* 1

parameters = [
    {'n_estimators': [100, 200],
    'max_depth':[6, 8, 10, 12],
    'min_samples_leaf': [1, 3, 5, 7],
    'min_samples_split': [1, 3, 5, 7],
    'n_jobs': [-1]}
]
#parameter 여러 개 넣으면 너무 느려짐...
#여기서 몇 개만 빼와서 파라미터 넣고 돌려보는 게 -> randomsearch

import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils.testing import all_estimators #testing 빼도 돌아감
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

#grid search import
from sklearn.model_selection import GridSearchCV


import warnings
warnings.filterwarnings('ignore')

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=44
)

#SVC모델에는 C, kernel, gamma라는 parameter: 알아서 공부하기
#default만 해도 85는 나온다
#kernel에 주는 값을 보니 대충 activation이랑 비슷한 개념이겠구나...

#파라미터만 해도 20번, cv는 5번(n_splits) -> 20*5 = fit 100번

#2. 모델
kfold = KFold(n_splits=5, shuffle=True) #몇 개로 조각낼 것인지(n_splits)
                                        #shuffle: 섞는다 즉 다섯으로 조각을 내서 섞겠다

# model = SVC()

#model이라고 꼭 이름 붙일 필요 없음. 자기 마음대로 붙여도 됨. 변수 이름임.
model = GridSearchCV(RandomForestRegressor(), parameters, cv=kfold, verbose=2) #괄호 안에 모델명. svc라는 모델을 girdserachcv로 쓰겠다

# scores = cross_val_score(model, x_train, y_train, cv=kfold) #검증한 score
# print(model, ": ", scores)


#3. 훈련
model.fit(x_train, y_train) #fit은 100번
#엥?? 그럼 여태까지 머신러닝은 죄다 fit을 한 번만 했던 건가??? 근데 그런 score가??

#4. 평가, 예측
#20번 중에 제일 좋은 조합을 반환해준다 
#estimator: 평가자
print("최적의 매개변수: ", model.best_estimator_)

y_predict = model.predict(x_test)
print("최종정답률: ", r2_score(y_test, y_predict))



'''
최적의 매개변수:  RandomForestRegressor(max_depth=12, min_samples_leaf=7, min_samples_split=5,
                      n_jobs=-1)
최종정답률:  0.4815785110805687
'''