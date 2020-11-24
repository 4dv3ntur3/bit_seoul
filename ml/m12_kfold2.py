#2020-11-24 (12일차)
#cross validation(cv): k-fold 모델별 비교 

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score #모델에 관여하는 kFold



#1. 데이터
iris = pd.read_csv('./data/csv/iris_ys.csv', header=0)
x = iris.iloc[:, :4]
y = iris.iloc[:, -1] #iloc[:, -1]

print(x.shape) #(150, 4)
print(y.shape) #(150,)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=50
)


#2. 모델
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

#모델 구성 전에
kfold = KFold(n_splits=5, shuffle=True) #몇 개로 조각낼 것인지(n_splits)
                                        #shuffle: 섞는다 즉 다섯으로 조각을 내서 섞겠다

model = LinearSVC()
# model = SVC()
# model = KNeighborsClassifier()
# model = KNeighborsRegressor()
# model = RandomForestClassifier()
# model = RandomForestRegressor() 


#model과 kfold를 엮어 줘야 함
scores = cross_val_score(model, x_train, y_train, cv=kfold) #검증한 score
print(model, "\nscores: ", scores)



# '''
# #3. 훈련
# model.fit(x_train, y_train)


# #4. 평가
# from sklearn.metrics import r2_score, accuracy_score

# score = model.score(x_test, y_test)  
# print("model.score: ", score)

# y_predict = model.predict(x_test)
# print(y_test[:10], "의 예측 결과: \n", y_predict[:10])

# metrics_score = accuracy_score(y_test, y_predict)
# print("accuracy_score: " , metrics_score)

# metrics_score = r2_score(y_test, y_predict)
# print("r2_score: ", metrics_score)

# '''



'''
LinearSVC()
scores:  [0.91666667 0.875      0.79166667 0.83333333 0.79166667]


SVC()
scores:  [1.         1.         1.         1.         0.91666667]


KNeighborsClassifier()
scores:  [1. 1. 1. 1. 1.]


KNeighborsRegressor()
scores:  [0.98846154 0.98796992 0.94810811 0.98662953 0.97530547]


RandomForestClassifier()
scores:  [1.         1.         1.         1.         0.95833333]


RandomForestRegressor()
scores:  [0.98864051 0.98430061 0.99904309 1.         0.99850235]
'''