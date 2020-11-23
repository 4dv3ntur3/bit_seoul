#2020-11-23 (11일차)
#wine.csv 이용해서 Deep Learning


import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler #robust = 이상치 제거에 효과적


#iris는 분류이므로 _classifier만 사용 (총 모델 4개)
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor 
                                                   #사이킷런의 이름을 보고 이해하는 능력 필요
                                                   #classifier: 분류 regressor: 회귀
                                                   #cf. logistic regressor는 regressor지만 분류

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor #feature importance
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical #sklearn의 one hot encoding은 0을 안 넣어 주고, keras의 것은 0부터 시작한다




#1. 데이터

data_np = np.loadtxt('./data/csv/winequality-white.csv', delimiter=';', skiprows=1) #head 제외하고 읽음


x = data_np[:, :data_np.shape[1]-1]
y = data_np[:, data_np.shape[1]-1:]
print(y.shape)


#one hot encoding
y = to_categorical(y)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=66, shuffle=True, train_size=0.7
)

scaler = StandardScaler()
scaler.fit(x_train)

scaler.transform(x_train)
scaler.transform(x_test)


print(x_train.shape)



#2. 모델(일단은 dafault만 미세 조정은 추후에)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(30, activation='relu', input_shape=(x_train.shape[1],))) #flatten하면서 곱하고 dense에서 또 100 곱함 
                                        #Conv2d의 activatio n default='relu'
                                        #LSTM의 activation default='tanh'
model.add(Dense(100, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(500, activation='relu'))
model.add(Dense(1000, activation='relu'))
model.add(Dense(800, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='softmax'))


#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping

es = EarlyStopping(monitor='loss', patience=100, mode='auto')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc']) #2진 분류면 one hot encoding 필요 없음. 3진 분류는 해야 됨
model.fit(x_train, y_train, batch_size=32, validation_split=0.2, epochs=1000)


#4. 평가, 예측
# y_predict = model.predict(x_test) #y_data가 나오는지 보면 된다 


# print(x_test, '의 예측 결과: ', y_predict)

acc_1 = model.evaluate(x_test, y_test)
print('model.evaluate: ', acc_1) 


# y_predict = model.predict(x_test)

# metrics_score = accuracy_score(y_test, y_predict)
# print("accuracy_score: " , metrics_score)

# # metrics_score = r2_score(y_test, y_predict)
# # print("r2_score: ", metrics_score)


# print(y_test[:10], "의 예측 결과: \n", y_predict[:10])



