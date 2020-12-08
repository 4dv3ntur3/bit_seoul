#2020-12-08
#SFM 해서 돌릴 때마다 모델이 생길 텐데 가장 좋은 놈만 남긴다
#주석으로 n이 몇 개일 때 가장 좋았고 뭘 남겼는지... 기록해 두기

#load_boston + pickle 

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

import pickle 

for thresh in thresholds:

    #########selectFromModel parameter 정리해 두기 
    selection = SelectFromModel(model, threshold=thresh, prefit=True) #thresh= 임계값

    #첫 번째 돌아갈 때는 그 첫번째 값 이상의 놈들을 돌려라
    #두 번째는 첫 번쨰 빼고 돌아감 

    #선택된 놈은 x_train으로 transform 하겠다 
    select_x_train = selection.transform(x_train)

    selection_model = XGBRegressor(n_jobs=6)
    selection_model.fit(select_x_train, y_train)

    select_x_test = selection.transform(x_test)
    y_predict = selection_model.predict(select_x_test)

    #score
    score = r2_score(y_test, y_predict)
    print("Thresh=%.3f, n=%d, R2: %.2F%%" % (thresh, select_x_train.shape[1],
          score*100.0))

    n = select_x_train.shape[1]

    pickle.dump(model, open("./save/xgb_save/boston-"+str(n)+"-pickle.dat", "wb"))
    print("저장 완료!")
    


'''
R2:  0.9221188544655419
[0.00134153 0.00363372 0.01203115 0.01220458 0.01447935 0.01479119
 0.0175432  0.03041655 0.04246345 0.0518254  0.06949984 0.30128643
 0.42848358]
Thresh=0.001, n=13, R2: 92.21%
저장 완료!
Thresh=0.004, n=12, R2: 92.16%
저장 완료!
Thresh=0.012, n=11, R2: 92.03%
저장 완료!
Thresh=0.012, n=10, R2: 92.19%
저장 완료!
Thresh=0.014, n=9, R2: 93.08%  -> 9만 남겨 놓겠다 
저장 완료!
Thresh=0.015, n=8, R2: 92.37%
저장 완료!
Thresh=0.018, n=7, R2: 91.48%
저장 완료!
Thresh=0.030, n=6, R2: 92.71%
저장 완료!
Thresh=0.042, n=5, R2: 91.74%
저장 완료!
Thresh=0.052, n=4, R2: 92.11%
저장 완료!
Thresh=0.069, n=3, R2: 92.52%
저장 완료!
Thresh=0.301, n=2, R2: 69.41%
저장 완료!
Thresh=0.428, n=1, R2: 44.98%
저장 완료!


'''