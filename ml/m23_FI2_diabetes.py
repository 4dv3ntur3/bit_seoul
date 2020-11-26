#2020-11-25
#feature importance:diabetes

#기준은 xbgoost
# 0. 디폴트 xgboost
# 1. feature importance=0인 것 제거. 압축 아님 or #2. 하위 30% 제거
# 실행 3번

#feature importance=0인 게 없으므로 하위 30%제거


import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier, XGBRegressor

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split



#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target #y값 보고 회귀인지 분류인지 판단. 단, 시계열의 경우 y값을 뭘로 할지는 내가 선택.


print(x.shape) #(442, 10)
print(y.shape) #(442,)


# x_1 = x[:, :2]
# x_2 = x[:, 3:5]
# x_3 = x[:, 6:9]

# x = np.concatenate([x_1, x_2, x_3], axis=1)



x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=66 #or cancer.data, cancer.target
)



#2. 모델 구성

#max_depth 바꿔 보기
model = XGBRegressor(max_depth=4)



#3. 훈련
model.fit(x_train, y_train)


print(model.feature_importances_) #column은 30개고, 각 column마다 중요도가 나온다
                                #   column값이 0인 것들은 필요 없는 column 
                                #   얘네까지 같이 돌리면 1. 속도저하 2. 자원량 낭비
                                #   따라서 0이 아닌 column만 모아서 돌려도 결과는 동일하다
                                #   단, 조건: accuracy_score를 신뢰할 수 있어야 함

# print(datasets.feature_names)
# print(x)

fi = model.feature_importances_
indices = np.argsort(fi)[::-1] #거꾸로 정렬(즉, 제일 작은 값이 뒤에 와 있다)

# print(indices)

# del_index = []
# for i in indices:
#     if i < 0.7*int(len(fi)): #하위 30퍼센트
#         del_index.append()


x_train = x_train[:, indices[:6]]
x_test = x_test[:, indices[:6]]

# print(x)

# print(indices)



#4. 평가 및 예측
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)



print("acc: ", acc) 







#feature importance
# import matplotlib.pyplot as plt
# import numpy as np
# def plot_feature_importances_cancer(model):
#     n_features = datasets.data.shape[1]
#     plt.barh(np.arange(n_features), model.feature_importances_,
#                 align='center')

#     plt.yticks(np.arange(n_features))
#     plt.xlabel("feature importances")
#     plt.ylabel("features")
#     plt.ylim(-1, n_features)



# plot_feature_importances_cancer(model)
# plt.show() #[0.01759811 0.02607087 0.6192673  0.33706376]







'''
default
acc:  0.31163770597265394
[0.03951401 0.08722725 0.18159387 0.08551976 0.04845208 0.06130722
 0.05748899 0.0561045  0.32311246 0.05967987]


하위 30퍼센트 제거
acc:  0.2568870532760785
'''
