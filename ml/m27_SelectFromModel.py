#2020-11-26
#XGBooster

from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np

#뭔가 모델을 선택할 것 같은 애
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score

x, y = load_boston(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state = 66
)


model = XGBRegressor(n_jobs=-1)
model.fit(x_train, y_train)

score = model.score(x_test, y_test)
print("R2: ", score)


#thresholds도 변수명
#feature_importance 가 오름차순으로 정렬된다
thresholds = np.sort(model.feature_importances_) 
print(thresholds)
# print(type(thresholds))

for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True) #thresh= 임계값

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
R2:  0.9221188544655419
[0.00134153 0.00363372 0.01203115 0.01220458 0.01447935 0.01479119
 0.0175432  0.03041655 0.04246345 0.0518254  0.06949984 0.30128643
 0.42848358]
'''


'''
n이 9개일 때 오히려 R2 점수가 높다
각 feature에 대해서 모델을 선택해 주는 놈!
더 편해졌다(어제까진 일일이 한땀한땀 했다)
단, 얘 역시도 100프로 신뢰 불가.
처음 위의 모델이 제대로 된 상태여야 한다(근데 그건 또 피쳐가 다 들어가 있는 상탠데...)
만약 두 개가 다 신뢰가 간다는 가정하에 9개 들어간 게 가장! (제일 약한 거 4개를 빼는 게 오히려 낫다)

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
'''


