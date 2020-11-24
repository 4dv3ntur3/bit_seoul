#2020-11-24 (12일차)
#pipeline + boston


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
iris = pd.read_csv('./data/csv/boston_house_prices.csv', header=1)

x = iris.iloc[:, :-1]
y = iris.iloc[:, -1:] 


x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=44
)



#scaling 엮기: pipeline-cv에서 과적합을 피하기 위해서 했음 
#scaler: 4개에 사실은 2개 더 있음 (이상치 제거 등등...)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler



parameters = [
    {'rfc__n_estimators': [100, 200],
    'rfc__max_depth':[2, 4, 6, 8, 10, 12, 20],
    'rfc__min_samples_leaf': [1, 3, 5, 7, 9],
    'rfc__min_samples_split': [1, 3, 5, 7],
    'rfc__n_jobs': [-1]}
]


#2. 모델
pipe = Pipeline([("scaler", MinMaxScaler()), ("rfc", RandomForestRegressor())])
model = RandomizedSearchCV(pipe, parameters, cv=5, verbose=2)


#3. 훈련
model.fit(x_train, y_train) #fit은 100번

print("최적의 매개변수: ", model.best_estimator_)

# y_predict = model.predict(x_test)
print("acc: ", model.score(x_test, y_test))
# print("최적의 매개변수: ", model.best_params_)



'''
[Parallel(n_jobs=1)]: Done  50 out of  50 | elapsed:   29.5s finished
최적의 매개변수:  Pipeline(steps=[('scaler', MinMaxScaler()),
                ('rfc',
                 RandomForestRegressor(max_depth=10, min_samples_split=5,
                                       n_estimators=200, n_jobs=-1))])
acc:  0.8852446818725208


최적의 매개변수:  Pipeline(steps=[('scaler', MinMaxScaler()),
                ('rfc',
                 RandomForestRegressor(max_depth=20, min_samples_split=5,
                                       n_jobs=-1))])
acc:  0.890408394226351

'''
