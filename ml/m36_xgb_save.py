#2020-12-08
#XGB 자체에서 제공하는 기능으로 저장


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



# 모델 & 가중치까지 같이 저장 
model.save_model('./save/xgb_save/cancer.xgb.model')
print("save complete!")


# 모델 불러오기
model2 = XGBClassifier() #model2 먼저 명시 
model2.load_model('./save/xgb_save/cancer.xgb.model')
print("load complete!")


#3. 훈련
model2.fit(x_train, y_train)

#4. 평가 및 예측
acc = model2.score(x_test, y_test)
print("acc2: ", acc)


'''
acc:  0.9736842105263158
save complete!
load complete!
acc2:  0.9736842105263158
'''