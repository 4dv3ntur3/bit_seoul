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


#기준 xg
#FI 0 제거
#하위 30% 제거
#디폴트와 성능비교

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRFRegressor
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_diabetes

x, y = load_diabetes(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, shuffle=True, train_size=0.8)


model1 = XGBRFRegressor()
model1.fit(x_train, y_train)

default_score =model1.score(x_test, y_test)


model = XGBRFRegressor()
model.fit(x_train, y_train)
print(model.feature_importances_) 

index7 =np.sort(model.feature_importances_)[::-1][int(0.7 *len(model.feature_importances_) )]

delete_list = []
for i in model.feature_importances_:
    if i < index7:
        print(i,"제거 ")
        delete_list.append(model.feature_importances_.tolist().index(i))



# print(delete_list)
model2 = XGBRFRegressor(max_depth=4)

# print(x_train.shape)
x_train  = np.delete(x_train, delete_list, axis=1)
x_test  = np.delete(x_test, delete_list, axis=1)
# print(x_train.shape)



'''

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target #y값 보고 회귀인지 분류인지 판단. 단, 시계열의 경우 y값을 뭘로 할지는 내가 선택.

# print(load_boston().feature_names)
# print(x.shape) #506, 13
# print(y.shape) #506

# dataset_pd = pd.DataFrame(x)
# dataset_pd = dataset_pd.iloc[:, ]


x_1 = x[:, :1]
x_2 = x[:, 2:9]
x_3 = x[:, 10:11]
x_4 = x[:, 12:]

x = np.concatenate([x_1, x_2, x_3, x_4], axis=1)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=66 #or cancer.data, cancer.target
)


#2. 모델 구성

#max_depth 바꿔 보기
model = XGBRegressor(max_depth=4)


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




0.9328109815565079


acc:  0.9148959596817177







(150, 4)
(150,)
acc:  0.8666666666666667


[0.01669537 0.00150525 0.02149532 0.0007204  0.05927434 0.29080436
 0.01197547 0.05330402 0.0360474  0.02261044 0.07038534 0.01352609
 0.40165624]


'''