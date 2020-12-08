#2020-12-08

#2020-12-08
#SFM 해서 돌릴 때마다 모델이 생길 텐데 가장 좋은 놈만 남긴다
#주석으로 n이 몇 개일 때 가장 좋았고 뭘 남겼는지... 기록해 두기

#load_breast_cancer + XGB.save_model

from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np

#뭔가 모델을 선택할 것 같은 애
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score

x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state = 66
)


model = XGBClassifier(n_jobs=-1)
model.fit(x_train, y_train)

score = model.score(x_test, y_test)
print("acc: ", score)


#thresholds도 변수명
#feature_importance 가 오름차순으로 정렬된다
thresholds = np.sort(model.feature_importances_) 
print(thresholds)
# print(type(thresholds))

import joblib 

for thresh in thresholds:

    #########selectFromModel parameter 정리해 두기 
    selection = SelectFromModel(model, threshold=thresh, prefit=True) #thresh= 임계값

    #첫 번째 돌아갈 때는 그 첫번째 값 이상의 놈들을 돌려라
    #두 번째는 첫 번쨰 빼고 돌아감 

    #선택된 놈은 x_train으로 transform 하겠다 
    select_x_train = selection.transform(x_train)

    selection_model = XGBClassifier(n_jobs=6)
    selection_model.fit(select_x_train, y_train)

    select_x_test = selection.transform(x_test)
    y_predict = selection_model.predict(select_x_test)

    #score
    score = accuracy_score(y_test, y_predict)
    print("Thresh=%.3f, n=%d, acc: %.2F%%" % (thresh, select_x_train.shape[1],
          score*100.0))

    n = select_x_train.shape[1]

    selection_model.save_model("./save/xgb_save/cancer-"+str(n)+"-xgb.model")
    print("저장 완료!")
    


'''
acc:  0.9736842105263158
[0.         0.         0.00037145 0.00233393 0.00278498 0.00281184
 0.00326043 0.00340272 0.00369179 0.00430626 0.0050556  0.00513449
 0.0054994  0.0058475  0.00639412 0.00769184 0.00775311 0.00903706
 0.01171023 0.0136856  0.01420499 0.01813928 0.02285903 0.02365488
 0.03333857 0.06629944 0.09745205 0.11586285 0.22248562 0.28493083]
Thresh=0.000, n=30, acc: 97.37%
저장 완료!
Thresh=0.000, n=30, acc: 97.37%
저장 완료!
Thresh=0.000, n=28, acc: 97.37%
저장 완료!
Thresh=0.002, n=27, acc: 97.37%
저장 완료!
Thresh=0.003, n=26, acc: 97.37%
저장 완료!
Thresh=0.003, n=25, acc: 97.37%
저장 완료!
Thresh=0.003, n=24, acc: 97.37%
저장 완료!
Thresh=0.003, n=23, acc: 97.37%
저장 완료!
Thresh=0.004, n=22, acc: 96.49%
저장 완료!
Thresh=0.004, n=21, acc: 96.49%
저장 완료!
Thresh=0.005, n=20, acc: 97.37%
저장 완료!
Thresh=0.005, n=19, acc: 97.37%
저장 완료!
Thresh=0.005, n=18, acc: 96.49%
저장 완료!
Thresh=0.006, n=17, acc: 96.49%
저장 완료!
Thresh=0.006, n=16, acc: 96.49%
저장 완료!
Thresh=0.008, n=15, acc: 97.37%
저장 완료!
Thresh=0.008, n=14, acc: 97.37%
저장 완료!
Thresh=0.009, n=13, acc: 98.25% 
저장 완료!
Thresh=0.012, n=12, acc: 98.25%
저장 완료!
Thresh=0.014, n=11, acc: 98.25%
저장 완료!
Thresh=0.014, n=10, acc: 98.25%
저장 완료!
Thresh=0.018, n=9, acc: 97.37%
저장 완료!
Thresh=0.023, n=8, acc: 97.37%
저장 완료!
Thresh=0.024, n=7, acc: 98.25% -> 얘만 남겨 두고 다 지우겠다 
저장 완료!
Thresh=0.033, n=6, acc: 97.37%
저장 완료!
Thresh=0.066, n=5, acc: 95.61%
저장 완료!
Thresh=0.097, n=4, acc: 96.49%
저장 완료!
Thresh=0.116, n=3, acc: 94.74%
저장 완료!
Thresh=0.222, n=2, acc: 91.23%
저장 완료!
Thresh=0.285, n=1, acc: 88.60%
저장 완료!
'''