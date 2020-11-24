#2020-11-24 (12일차)
#Decision Tree: 의사결정트리
#feature importance

#모델 구조에 대한 설명을 여태 하나도 안 했음. 하지만 얘는 해야 한다.
#randomforest: 의사결정트리를 랜덤하게 모아서 (ensemble에서 import했음)
#decision tree를 boosting -> XGBooster(XGBooster, AdaBoost, LGBM: 다 트리구조)
#트리 구조의 모델들은 현재 사이킷런(머신러닝 쪽)에서 성능이 가장 좋다

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
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
model = DecisionTreeClassifier(max_depth=4)

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


'''
0.9122807017543859
[0.         0.0624678  0.         0.         0.         0.
 0.         0.         0.         0.         0.00492589 0.
 0.         0.02036305 0.         0.         0.         0.
 0.         0.02364429 0.         0.01695087 0.         0.75156772
 0.         0.         0.         0.12008039 0.         0.        ]
'''


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
