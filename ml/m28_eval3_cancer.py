#2020-12-07
#cancer: 2진분류
#eval_metric: auc or logloss or error


import numpy as np

from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score


#1. 데이터

# dataset = load_boston()
# x = dataset.data
# y = dataset.target

x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, 
    y, 
    train_size=0.8, 
    random_state=77,
    shuffle=True
)


#2. 모델
# model = XGBRegressor(n_estimators=1000, learning_rate=0.1)

#estimator 없이 
#default가 몇 갤까? -> 100번 돈다 : n_estimator default 100
model = XGBClassifier(n_estimators=500, learning_rate=0.1, max_depth=20)


#3. 훈련
model.fit(x_train, y_train, verbose=1, #다 보여 준다(0 / 1 , False / True)

    eval_metric='error', #keras의 metrics와 동일. RMSE를 쓰겠다. 
    # eval_set=[(x_test, y_test)] #평가는 x_test, y_test & 지표는 RMSE(MSE에 루트) 
    eval_set=[(x_train, y_train), (x_test, y_test)] #metrics는 어차피 훈련에 반영되지 않으니까 훈련 set에 대해서도 metric 볼 수 있다 

    #출력은 n_estimators만큼
)

# score = model.score(x_test, y_test)



#4. 평가 및 예측 
#XGBoost의 대표적 평가 지표 5개
#대회에서 평가 지표 제공해 줬는데 이해를 못 하겠다 -> 일단 유사 지표 넣어서 달림 
#RMSE, MAE, logloss, error(acc와 유사), auc(이진분류)

#평가 결과
#evalate와 비슷하지만 결과치가 출력되기 때문에 조금 난해하게 보인다 
results = model.evals_result()
print("eval's results: ", results)

y_pred = model.predict(x_test)

score = model.score(x_test, y_test)
acc = accuracy_score(y_pred, y_test)

print("acc: ", acc)
print("score: ", score)


'''
error / auc / logloss
acc:  0.9649122807017544
'''