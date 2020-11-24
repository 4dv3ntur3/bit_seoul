#2020-11-24 (12일차)
#feature importance
#Decision Tree: RF

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
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
model = RandomForestClassifier(max_depth=4)

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
acc:  0.956140350877193
[0.033663   0.01043803 0.05062287 0.016819   0.00299496 0.0131482
 0.04759229 0.08018194 0.00253741 0.00271297 0.00615898 0.00258221
 0.01756898 0.04321753 0.00183673 0.00340163 0.00234316 0.00237367
 0.00394806 0.00339268 0.18182102 0.01898861 0.13915023 0.13859176
 0.0080163  0.00861859 0.04236554 0.10196094 0.00649825 0.00645444]

m18_1_decisionTree.py의 것과 똑같은가?
아님! randomforest에서 바뀌었다 -> 얘의 feature_importance가 더 신뢰가 간다
그러므로 기준치를 잡아서 미달되는 feature를 뺀다
ex) 0.1이하 다 뺀다 -> 90퍼센트 이상의 중요도 있는 놈만 쓰겠다
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
