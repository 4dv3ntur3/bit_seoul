#2020-11-25
#feature importance:iris
#column이 4개라서 별 의미는 없을 것으로 예상

#기준은 xbgoost
# 0. 디폴트 xgboost
# 1. feature importance=0인 것 제거. 압축 아님 or #2. 하위 30% 제거
#B ZN CHAS OUT 
# 실행 3번


import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier, XGBRegressor

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split



#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target #y값 보고 회귀인지 분류인지 판단. 단, 시계열의 경우 y값을 뭘로 할지는 내가 선택.

# print(load_boston().feature_names)
# print(x.shape) #506, 13
# print(y.shape) #506

# dataset_pd = pd.DataFrame(x)
# dataset_pd = dataset_pd.iloc[:, ]



x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=66 #or cancer.data, cancer.target
)


#2. 모델 구성

#max_depth 바꿔 보기
model = XGBRegressor(max_depth=4)


#3. 훈련
model.fit(x_train, y_train)

#4-1. 평가 및 예측
acc = model.score(x_test, y_test)

print("acc: ", acc) 

print(model.feature_importances_) #column은 30개고, 각 column마다 중요도가 나온다
                                  #column값이 0인 것들은 필요 없는 column 
                                  #얘네까지 같이 돌리면 1. 속도저하 2. 자원량 낭비
                                  #따라서 0이 아닌 column만 모아서 돌려도 결과는 동일하다
                                  #단, 조건: accuracy_score를 신뢰할 수 있어야 함




#4-2. FI 적용
fi = model.feature_importances_
indices = np.argsort(fi)[::-1]

print("FI 내림차순 정렬: ", indices)

del_index = []
for i in indices:
    if i < int(0.7*len(fi)):
        del_index.append(i)

print("삭제할 columns: ", del_index)


x_train = x_train[:, del_index]
x_test = x_test[:, del_index]



model.fit(x_train, y_train)
acc = model.score(x_test, y_test)


print("acc: ", acc) 





# #feature importance
# import matplotlib.pyplot as plt
# import numpy as np
# def plot_feature_importances_cancer(model):
#     n_features = datasets.data.shape[1]
#     plt.barh(np.arange(n_features), model.feature_importances_,
#                 align='center')

#     plt.yticks(np.arange(n_features), datasets.feature_names)
#     plt.xlabel("feature importances")
#     plt.ylabel("features")
#     plt.ylim(-1, n_features)

# plot_feature_importances_cancer(model)
# plt.show() #[0.01759811 0.02607087 0.6192673  0.33706376]



'''
1. default
acc:  0.9328109815565079
[0.01669537 0.00150525 0.02149532 0.0007204  0.05927434 0.29080436
 0.01197547 0.05330402 0.0360474  0.02261044 0.07038534 0.01352609
 0.40165624]

 2. 하위 30퍼센트 제거
FI 내림차순 정렬:  [12  5 10  4  7  8  9  2  0 11  6  1  3]
삭제할 columns:  [5, 4, 7, 8, 2, 0, 6, 1, 3]
acc:  0.8555379534091828


'''


'''
무식한 방법
x_1 = x[:, :1]
x_2 = x[:, 2:9]
x_3 = x[:, 10:11]
x_4 = x[:, 12:]

x = np.concatenate([x_1, x_2, x_3, x_4], axis=1)

'''