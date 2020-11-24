#2020-11-24 (12일차)
#pipeline + wine
#scale 안 한 wine과 비교 


import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators #testing 빼도 돌아감
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

#grid search import
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


import warnings
warnings.filterwarnings('ignore')

#1. 데이터
iris = pd.read_csv('./data/csv/winequality-white.csv', sep=';')


x = iris.iloc[:, :-1]
y = iris.iloc[:, -1:]



# print(x.shape) #4898, 0

# print(y.shape) #4898, 1


x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=35
)


#scaling 엮기: pipeline-cv에서 과적합을 피하기 위해서 했음 
#scaler: 4개에 사실은 2개 더 있음 (이상치 제거 등등...)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler



parameters = [
    {'rfc__n_estimators': [100, 200],
    'rfc__max_depth':[6, 8, 10, 12, 20, 30],
    'rfc__min_samples_leaf': [1, 3, 5, 7, 9],
    'rfc__min_samples_split': [1, 3, 5, 7, 9, 11],
    'rfc__n_jobs': [-1]}
]


#2. 모델
pipe = Pipeline([("scaler", StandardScaler()), ("rfc", RandomForestClassifier())])
model = RandomizedSearchCV(pipe, parameters, cv=5, verbose=2)


#3. 훈련
model.fit(x_train, y_train) #fit은 100번

# print("최적의 매개변수: ", model.best_estimator_)
print("최적의 매개변수: ", model.best_params_)

# y_predict = model.predict(x_test)
print("acc: ", model.score(x_test, y_test))


'''
최적의 매개변수:  Pipeline(steps=[('scaler', MinMaxScaler()),
                ('rfc',
                 RandomForestClassifier(max_depth=20, min_samples_leaf=3,
                                        min_samples_split=7, n_estimators=200,
                                        n_jobs=-1))])
acc:  0.6571428571428571


최적의 매개변수:  Pipeline(steps=[('scaler', StandardScaler()),
                ('rfc',
                 RandomForestClassifier(max_depth=12, min_samples_split=5,
                                        n_estimators=200, n_jobs=-1))])
acc:  0.6438775510204081
최적의 매개변수:  {'rfc__n_jobs': -1, 'rfc__n_estimators': 200, 'rfc__min_samples_split': 5, 'rfc__min_samples_leaf': 1, 'rfc__max_depth': 12}

[Parallel(n_jobs=1)]: Done  50 out of  50 | elapsed:   24.5s finished
최적의 매개변수:  {'rfc__n_jobs': -1, 'rfc__n_estimators': 200, 'rfc__min_samples_split': 7, 'rfc__min_samples_leaf': 1, 'rfc__max_depth': 30}
acc:  0.6683673469387755

'''

#출력되지 않는 거는 default 값(리스트로 준 첫 번째 값)으로 됐기 때문