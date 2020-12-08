#2020-12-08 
#XGBoost 모델 저장 / 불러오기
#pickle (파이썬에서 제공함) -> 다른 모델들도 이걸로 저장할 수 있고 다른 확장자도 가능하다 

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split



#1. 데이터
cancer = load_breast_cancer()
x = cancer.data
y = cancer.target #y값 보고 회귀인지 분류인지 판단. 단, 시계열의 경우 y값을 뭘로 할지는 내가 선택.

print(x.shape) #569, 30
print(y.shape) #569, 

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=66 #or cancer.data, cancer.target
)

#2. 모델 구성
model = XGBClassifier(max_depth=4)


#3. 훈련
model.fit(x_train, y_train)

#4. 평가 및 예측
acc = model.score(x_test, y_test)
print("acc: ", acc) #acc:  0.9736842105263158



# 모델 저장 (pickle.dump) 가중치까지 같이 저장 
import pickle
pickle.dump(model, open("./save/xgb_save/cancer.pickle.dat", "wb"))
print("save complete!")


# 모델 불러오기 (pickle.load)
model2 = pickle.load(open("./save/xgb_save/cancer.pickle.dat", "rb"))
print("load complete!")


#3. 훈련
model2.fit(x_train, y_train)

#4. 평가 및 예측
acc = model2.score(x_test, y_test)
print("acc2: ", acc)


'''
동일하게 잘 나온다! 가중치 save & load 제대로 됐다 
acc:  0.9736842105263158
save complete!
load complete!
acc2:  0.9736842105263158
'''