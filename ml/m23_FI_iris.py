#2020-11-25
#feature importance:iris
#column이 4개라서 별 의미는 없을 것으로 예상

#기준은 xbgoost
# 0. 디폴트 xgboost
# 1. feature importance=0인 것 제거. 압축 아님 or #2. 하위 30% 제거
# 실행 3번



#[0.01759811 0.02607087 0.6192673  0.33706376]
#맨 마지막 column out

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split






#1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target #y값 보고 회귀인지 분류인지 판단. 단, 시계열의 경우 y값을 뭘로 할지는 내가 선택.

print(x.shape) 
print(y.shape) 

x = x[:, :3]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=66 #or cancer.data, cancer.target
)


#2. 모델 구성

#max_depth 바꿔 보기
model = XGBClassifier(max_depth=4)


#3. 훈련
model.fit(x_train, y_train)

#4. 평가 및 예측
acc = model.score(x_test, y_test)

print("acc: ", acc) 
# print(model.feature_importances_) #column은 30개고, 각 column마다 중요도가 나온다
#                                   #column값이 0인 것들은 필요 없는 column 
#                                   #얘네까지 같이 돌리면 1. 속도저하 2. 자원량 낭비
#                                   #따라서 0이 아닌 column만 모아서 돌려도 결과는 동일하다
#                                   #단, 조건: accuracy_score를 신뢰할 수 있어야 함



#feature importance
import matplotlib.pyplot as plt
import numpy as np
def plot_feature_importances_cancer(model):
    n_features = datasets.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_,
                align='center')

    plt.yticks(np.arange(n_features), datasets.feature_names)
    plt.xlabel("feature importances")
    plt.ylabel("features")
    plt.ylim(-1, n_features)

plot_feature_importances_cancer(model)
plt.show() #[0.01759811 0.02607087 0.6192673  0.33706376]



'''
#0.9122807017543859


(150, 4)
(150,)
acc:  0.8666666666666667
'''