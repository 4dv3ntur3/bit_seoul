#2020-11-24 (12일차)
#randomsearch


import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators #testing 빼도 돌아감
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

#명칭 정확하게 기억할 것 
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


import warnings
warnings.filterwarnings('ignore')

#1. 데이터
iris = pd.read_csv('./data/csv/iris_ys.csv', header=0)

x = iris.iloc[:, :4]
y = iris.iloc[:, -1]


x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=44
)

#SVC모델에는 C, kernel, gamma라는 parameter: 알아서 공부하기
#default만 해도 85는 나온다
#kernel에 주는 값을 보니 대충 activation이랑 비슷한 개념이겠구나...

parameters = [
    {'C': [1, 10, 100, 1000], "kernel":["linear"]}, # C*kernel = 4*1 = 4
    {'C': [1, 10, 100, 1000], "kernel":["rbf"], "gamma":[0.001, 0.0001]}, #4*1*2
    {'C': [1, 10, 100, 1000], "kernel":["sigmoid"], "gamma":[0.001, 0.0001]} #4*1*2
]
#파라미터만 해도 20번, cv는 5번(n_splits) -> 20*5 = fit 100번

#2. 모델
kfold = KFold(n_splits=5, shuffle=True) #몇 개로 조각낼 것인지(n_splits)
                                        #shuffle: 섞는다 즉 다섯으로 조각을 내서 섞겠다

# model = SVC()

#model이라고 꼭 이름 붙일 필요 없음. 자기 마음대로 붙여도 됨. 변수 이름임.
model = RandomizedSearchCV(SVC(), parameters, cv=kfold, verbose=2) #괄호 안에 모델명. svc라는 모델을 girdserachcv로 쓰겠다
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
print("최종정답률: ", accuracy_score(y_test, y_predict))


#randomizedsearchcv는 몇 번 돌릴까? (몇프로 덜하는지. grid서치보다 얼마만큼 적게하는지)


'''
최적의 매개변수:  SVC(C=10, gamma=0.001)
최종정답률:  0.9666666666666667
'''
'''
[Parallel(n_jobs=1)]: Done  50 out of  50 | elapsed:    0.3s finished
최적의 매개변수:  SVC(C=10, kernel='linear')
최종정답률:  1.0
'''

# Fitting 5 folds for each of 10 candidates, totalling 50 fits