#2020-11-25 (13일차)
#diabetes: xgb
#과적합 방지
#1. 훈련데이터량을 늘린다
#2. 피쳐수를 줄인다
#3. regularization


#다한 사람은 모델을 완성해서 결과 주석으로 적어놓을 것 
from xgboost import XGBClassifier, XGBRegressor, plot_importance
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

parameters = [
    {"xgb__n_estimators": [100, 200, 300], "xgb__learning_rate": [0.1, 0.3, 0.001, 0.01], "xgb__max_depth":[4, 5, 6]},
    {"xgb__n_estimators": [90, 100, 110], "xgb__learning_rate": [0.1, 0.001, 0.01], "xgb__max_depth":[4, 5, 6], "xgb__colsample_bytree": [0.6, 0.9, 1]},
    {"xgb__n_estimators": [90, 110], "xgb__learning_rate": [0.1, 0.001, 0.5], "xgb__max_depth":[4, 5, 6], "xgb__colsample_bytree":[0.6, 0.9, 1],
    "xgb__colsample": [0.6, 0.7, 0.9]}
]


#score로 성능 비교
from sklearn.model_selection import RandomizedSearchCV, KFold, cross_val_score

datasets = load_diabetes()
x = datasets.data
y = datasets.target


x_train, x_test, y_train, y_test = train_test_split(
    datasets.data, datasets.target, random_state=66, shuffle=True, train_size=0.8 
)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline

pipe = Pipeline([("scaler", MinMaxScaler()), ("xgb", XGBRegressor())])

kfold = KFold(n_splits=5, shuffle=True)
model = RandomizedSearchCV(pipe, parameters, cv=kfold, verbose=2)

model.fit(x_train, y_train)
print("최적의 매개변수: ", model.best_estimator_)



from sklearn.metrics import r2_score
score = model.score(x_test, y_test)     #회귀에서는 r2_score = score이다
                                        #accuracy_score != score (둘이 같다는 건 분류에서나 통하는 이야기)
print("model.score: ", score)

'''
최적의 매개변수:  Pipeline(steps=[('scaler', MinMaxScaler()),
                ('xgb',
                 XGBRegressor(base_score=0.5, booster='gbtree', colsample=0.7,
                              colsample_bylevel=1, colsample_bynode=1,
                              colsample_bytree=0.6, gamma=0, gpu_id=-1,
                              importance_type='gain',
                              interaction_constraints='', learning_rate=0.1,
                              max_delta_step=0, max_depth=6, min_child_weight=1,
                              missing=nan, monotone_constraints='()',
                              n_estimators=110, n_jobs=0, num_parallel_tree=1,
                              random_state=0, reg_alpha=0, reg_lambda=1,
                              scale_pos_weight=1, subsample=1,
                              tree_method='exact', validate_parameters=1,
                              verbosity=None))])
model.score:  0.3151980691075743
'''
