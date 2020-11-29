#2020-11-24 (12일차)
#feature importance
#Decision Tree: XGB(eXtreme Gradient Boost)

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

#max_depth 바꿔 보기
# model = DecisionTreeClassifier(max_depth=4) #4번까지 잘랐다는 뜻
# model = RandomForestClassifier(max_depth=4)
# model = GradientBoostingClassifier(max_depth=4)
model = XGBClassifier(max_depth=4)


#3. 훈련
model.fit(x_train, y_train)

#4. 평가 및 예측
acc = model.score(x_test, y_test)
print("acc: ", acc) #0.9122807017543859
print(model.feature_importances_) #column은 30개고, 각 column마다 중요도가 나온다
                                  #column값이 0인 것들은 필요 없는 column 
                                  #얘네까지 같이 돌리면 1. 속도저하 2. 자원량 낭비
                                  #따라서 0이 아닌 column만 모아서 돌려도 결과는 동일하다
                                  #단, 조건: accuracy_score를 신뢰할 수 있어야 함




import matplotlib.pyplot as plt
import numpy as np
def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_,
                align='center')

    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("feature importances")
    plt.ylabel("features")
    plt.ylim(-1, n_features)

plot_feature_importances_cancer(model)
plt.show()






'''
(569, 30)
(569,)
acc:  0.9736842105263158
[0.         0.03518598 0.00053468 0.02371635 0.00661651 0.02328466
 0.00405836 0.09933352 0.00236719 0.         0.01060954 0.00473884
 0.01074011 0.01426315 0.0022232  0.00573987 0.00049415 0.00060479
 0.00522006 0.00680739 0.01785728 0.0190929  0.3432317  0.24493258
 0.00278067 0.         0.01099805 0.09473949 0.00262496 0.00720399]
'''