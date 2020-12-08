#2020-12-08
#n_jobs의 비밀 = 스레드? 코어? 
#n_jobs=-1이 항상 최적은 아닐 수도 있다. (다 쓰는 게 늘 옳은 것은 아님)


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

import time #시간 측정할 때 보여주는 것(출력하는 것)도 그만큼 delay 잡힘 

start = time.time()
for thresh in thresholds:

    #########selectFromModel parameter 정리해 두기 
    selection = SelectFromModel(model, threshold=thresh, prefit=True) #thresh= 임계값

    #첫 번째 돌아갈 때는 그 첫번째 값 이상의 놈들을 돌려라
    #두 번째는 첫 번쨰 빼고 돌아감 

    #선택된 놈은 x_train으로 transform 하겠다 
    select_x_train = selection.transform(x_train)

    selection_model = XGBRegressor(n_jobs=-1)
    selection_model.fit(select_x_train, y_train)

    select_x_test = selection.transform(x_test)
    y_predict = selection_model.predict(select_x_test)

    #score
    score = r2_score(y_test, y_predict)
    # print("Thresh=%.3f, n=%d, R2: %.2F%%" % (thresh, select_x_train.shape[1], score*100.0))


start2 = time.time()
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
    # print("Thresh=%.3f, n=%d, R2: %.2F%%" % (thresh, select_x_train.shape[1],
    #       score*100.0))

end = start2 - start
print("그냥 걸린 시간: ", end)

#현재시간 - start2
end2 = time.time() - start2
print("n_jobs 걸린 시간: ", end2)


#지금 PC의 코어가 6개인데 n_jobs=-1이면 코어를 다 쓰겠다는 뜻
#지금 PC 6코어 12스레드임 (intel i7-8)

'''
=====제일 빠르다===== n_jobs=-1은 구라였다. 6이 제일 빨랐다... 
그냥 걸린 시간:  2.817465305328369 / n_jobs=-1
n_jobs 걸린 시간:  1.6894824504852295 / n_jobs=6


그냥 걸린 시간:  2.6499135494232178 / n_jobs=-1
n_jobs 걸린 시간:  2.6788370609283447 / n_jobs=12

print문 없애고 
그냥 걸린 시간:  2.729222536087036 / n_jobs=-1
n_jobs 걸린 시간:  2.58608341217041


그냥 걸린 시간:  2.7237162590026855 / n_jobs=-1
n_jobs 걸린 시간:  2.0485219955444336 / n_jobs=8

'''
