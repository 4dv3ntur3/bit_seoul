#2020-11-25
#feature importance:iris
#column이 4개라서 별 의미는 없을 것으로 예상

#기준은 xbgoost
# 0. 디폴트 xgboost
# 1. feature importance=0인 것 제거. 압축 아님 or #2. 하위 30% 제거
# 0인 것 없으므로 하위 30% 제거
# 실행 3번


import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier, XGBRegressor


from sklearn.model_selection import train_test_split

wine=pd.read_csv('./data/csv/winequality-white.csv', sep=';', header=0)
y=wine['quality'].values
x=wine.drop('quality', axis=1).values #이후에 값 분포에 따라 재편성하면 acc 더 오른다 



print("x.shape: ", x.shape, "x.type: ", type(x)) #(4898, 11)
print(y.shape) #(4898,)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8)


# x_1 = x[:, :4]
# x_2 = x[:, 5:7]
# x_3 = x[:, 8:10]
# x_4 = x[:, 11:]

# x = np.concatenate([x_1, x_2, x_3, x_4], axis=1)




#2. 모델 구성
model = XGBClassifier(max_depth=4)


#3. 훈련
model.fit(x_train, y_train)

#4. 평가 및 예측
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

#pandas에는 slicing 기능 없어서 numpy로 바꿔 준 후 진행해야 한다 
#TypeError: '(slice(None, None, None), [1, 5, 3, 2, 0, 4, 6])' is an invalid key
x_train = x_train[:, del_index]
x_test = x_test[:, del_index]



model.fit(x_train, y_train)
acc = model.score(x_test, y_test)


print("acc: ", acc) 



# feature importance
# import matplotlib.pyplot as plt
# import numpy as np
# def plot_feature_importances_cancer(model):
#     n_features = x.shape[1]
#     plt.barh(np.arange(n_features), model.feature_importances_,
#                 align='center')

#     plt.yticks(np.arange(n_features))
#     plt.xlabel("feature importances")
#     plt.ylabel("features")
#     plt.ylim(-1, n_features)

# plot_feature_importances_cancer(model)
# plt.show()




'''
1. default 
acc:  0.6326530612244898
[0.06921455 0.12053432 0.07043804 0.08099263 0.06764299 0.09611965
 0.06570112 0.06899329 0.06871986 0.06973901 0.2219046 ]

2. 하위 30% 제거
FI 내림차순 정렬:  [10  1  5  3  7  0  9  2  6  4  8]
삭제할 columns:  [1, 5, 3, 0, 2, 6, 4]
acc:  0.6061224489795919
 '''