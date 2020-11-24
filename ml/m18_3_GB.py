#2020-11-24 (12일차)
#feature importance
#Decision Tree: GB(Gradient Boost)

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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
model = GradientBoostingClassifier(max_depth=4)


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
[1.35358043e-04 5.64470322e-02 9.98607465e-04 2.19969484e-03
 2.15118236e-03 1.32470960e-03 9.45408389e-04 7.83203338e-02
 1.21176813e-03 4.68284706e-05 5.17183553e-03 2.12956956e-04
 2.53155579e-03 1.24395854e-02 5.56194244e-03 3.08491592e-03
 6.26458931e-04 4.80248878e-04 2.41760744e-04 2.79146902e-03
 2.44010400e-01 3.41383303e-02 3.55672726e-03 4.13371404e-01   #얘가 제일 좋은 애(중요한 애)
 2.38120776e-03 9.74742648e-04 5.52372255e-03 1.17983820e-01
 1.74515839e-04 9.61476653e-04]

이거 역시도 
제대로 튠이 된 DT/ RF / GB 어떤 것을 쓸 것인지
feature를 가지고 장난질 칠 수 있다: feature engineering(중요도 낮은 거 다 솎아 버리고 잘라서 붙이고 loc, iloc....)
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
