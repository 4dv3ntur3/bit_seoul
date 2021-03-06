#2020-11-26
#실습
#####1. 
#모델에 그리드서치 또는 랜덤서치 적용
#최적의 R2값과 feature importance 구할 것

#####2. 
#위 스레드값으로 selectFromModel을 구해서 최적의 feature 개수를 구할 것
#위 feature의 개수로 데이터(feature)를 수정(삭제)해서 그리드서치 또는 랜덤서치 적용
#최적의 R2값을 구할 것

#1번 vs 2번

from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

parameters = [
    {"n_estimators": [100, 200, 300], "learning_rate": [0.1, 0.3, 0.001, 0.01], "max_depth":[4, 5, 6]},
    {"n_estimators": [90, 100, 110], "learning_rate": [0.1, 0.001, 0.01], "max_depth":[4, 5, 6], "colsample_bytree": [0.6, 0.9, 1]},
    {"n_estimators": [90, 110], "learning_rate": [0.1, 0.001, 0.5], "max_depth":[4, 5, 6], "colsample_bytree":[0.6, 0.9, 1],
    "colsample": [0.6, 0.7, 0.9]}
]

#뭔가 모델을 선택할 것 같은 애
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score

from sklearn.model_selection import RandomizedSearchCV, KFold



x, y = load_boston(return_X_y=True)
print(x.shape) #506, 13



x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state = 66
)


#randomized search
kfold = KFold(n_splits=5, shuffle=True)
model = RandomizedSearchCV(XGBRegressor(), parameters, cv=kfold) #, verbose=2)


model.fit(x_train, y_train)
print("최적의 매개변수: ", model.best_estimator_)


# model_xgb = XGBRegressor()
# model_xgb.fit(x_train, y_train)


#feature importances
fi = model.best_estimator_.feature_importances_
print("feature importances: ", fi)

thresholds = np.argsort(fi)[::-1]
print(thresholds)
x_train = x_train[:, thresholds[:9]]
x_test = x_test[:, thresholds[:9]]

model.fit(x_train, y_train)

from sklearn.metrics import r2_score
score = model.score(x_test, y_test)
print("R2: ", score)





'''
#thresholds도 변수명
#feature_importance 가 오름차순으로 정렬된다
thresholds = np.sort(model_xgb.feature_importances_) 
print("threshold: ", thresholds)

# print(type(thresholds))


for thresh in thresholds:
    selection = SelectFromModel(model_xgb, threshold=thresh, prefit=True) #thresh= 임계값

    #선택된 놈은 x_train으로 transform 하겠다 
    select_x_train = selection.transform(x_train)

    selection_model = XGBRegressor(n_jobs=-1)
    selection_model.fit(select_x_train, y_train)

    select_x_test = selection.transform(x_test)
    y_predict = selection_model.predict(select_x_test)

    #score
    score = r2_score(y_test, y_predict)
    print("Thresh=%.3f, n=%d, R2: %.2F%%" % (thresh, select_x_train.shape[1],
          score*100.0))


#모델을 한 번 돌리고, feature_importance_를 뽑은 다음에 feature 횟수만큼 또 돌려서 찾아내는 것
#따라서 처음 한 번 돌려서 중요도 뽑을 때가 중요하다
'''



'''
최적의 매개변수:  XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,       
             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.1, max_delta_step=0, max_depth=5,
             min_child_weight=1, missing=nan, 
monotone_constraints='()',
             n_estimators=300, n_jobs=0, num_parallel_tree=1, random_state=0,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
             tree_method='exact', validate_parameters=1, verbosity=None)

feature importances:  [0.01447935 0.00363372 0.01479119 0.00134153 0.06949984 0.30128643    
 0.01220458 0.0518254  0.0175432  0.03041655 0.04246345 0.01203115
 0.42848358]


R2:  0.9360567891784239

threshold:  [0.00134153 0.00363372 0.01203115 
0.01220458 0.01447935 0.01479119
 0.0175432  0.03041655 0.04246345 0.0518254  0.06949984 0.30128643
 0.42848358]

Thresh=0.001, n=13, R2: 92.21%
Thresh=0.004, n=12, R2: 92.16%
Thresh=0.012, n=11, R2: 92.03%
Thresh=0.012, n=10, R2: 92.19%
Thresh=0.014, n=9, R2: 93.08%
Thresh=0.015, n=8, R2: 92.37%
Thresh=0.018, n=7, R2: 91.48%
Thresh=0.030, n=6, R2: 92.71%
Thresh=0.042, n=5, R2: 91.74%
Thresh=0.052, n=4, R2: 92.11%
Thresh=0.069, n=3, R2: 92.52%
Thresh=0.301, n=2, R2: 69.41%
Thresh=0.428, n=1, R2: 44.98%

-> n=9일때가 최대


2) thread만큼 feature slicing
R2:  0.936352464493751





'''