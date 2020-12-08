#2020-12-08
#SFM 해서 돌릴 때마다 모델이 생길 텐데 가장 좋은 놈만 남긴다
#주석으로 n이 몇 개일 때 가장 좋았고 뭘 남겼는지... 기록해 두기

#iris + joblib

from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

#뭔가 모델을 선택할 것 같은 애
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score

x, y = load_iris(return_X_y=True)

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

    joblib.dump(model, "./save/xgb_save/iris-"+str(n)+"-joblib.dat")
    print("저장 완료!")
    

'''
acc:  0.9
[0.01759811 0.02607087 0.33706376 0.6192673 ]
Thresh=0.018, n=4, acc: 90.00%
저장 완료!
Thresh=0.026, n=3, acc: 90.00%
저장 완료!
Thresh=0.337, n=2, acc: 96.67% -> n=2 인 것만 남겨 놓겠다 
저장 완료!
Thresh=0.619, n=1, acc: 93.33%
저장 완료!
'''
