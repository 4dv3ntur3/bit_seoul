#2020-11-13 (5일차)
#데이터 전처리(Data Preprocessing): MinMax / Standard / Robust / MinAbs Scaler

##### 데이터 전처리: 아무리 튜닝을 잘해도 데이터가 구리면 안 좋은 결과가 나온다

# 부동소수점 연산이 더 수월
# 이 데이터를 최댓값으로 나누면 모든 데이터가 0~1 사이에 있게 된다
# 하지만 이건 데이터 조작이 아닌지...


# X는 훈련시키기 위한 데이터, 실제로는 Y값(타겟값)이 나와야 한다
# 따라서 Y는 건드리지 않고 X만 데이터를 압축시켜 놓음. 그래도 Y는 그대로! (y값은 그대로 라벨링)
# 0.001, 0.002, 0.003이 됐다고 해도 -> 4 임


from numpy import array
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Input

#1. 데이터

x = array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6],
          [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10],
          [9, 10, 11], [10, 11, 12],
          [2000, 3000, 4000], [3000, 4000, 5000], [4000, 5000, 6000],
          [100, 200, 300]])

y = array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 5000, 6000, 7000, 400])


# x_predict2 = array([6600, 6700, 6800])



#fit하고 transform
#1. MinMaxScaler
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler



from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7)

# x = scaler.transform(x)
# x = x.reshape(x.shape[0], x.shape[1], 1)



######### 01. MinMax 
# X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
# X_scaled = X_std * (max - min) + min

scaler = MinMaxScaler()
# loss, mse:  83646.0546875 83646.0546875
# 예측값:  [[110.70799]]


######### 02. Standard
# StandardScalar는 각 특성의 평균을 0, 분산을 1로 변경하여 모든 특성이 같은 크기를 가지게 함. 
# 특성의 최솟값과 최댓값 크기를 제한하지 않음

# # The standard score of a sample x is calculated as:
# z = (x - u) / s
# u is the mean of the training samples or zero if with_mean=False, and 
# s is the standard deviation of the training samples or one if with_std=False.


# scaler = StandardScaler()
# loss, mse:  10.600058555603027 10.600058555603027
# 예측값:  [[150.67647]




######### 03. Robust
# Nomalizer는 uclidian의 길이가 1이 되도록 데이터 포인트를 조정 ==> 각도가 많이 중요할 때 사용
# RobustScaler는 특성들이 같은 스케일을 갖게 되지만 평균대신 중앙값을 사용 ==> 극단값에 영향을 받지 않음


# scaler = RobustScaler()
# loss, mse:  10572.912109375 10572.912109375
# 예측값:  [[132.50508]]


######### 04. MaxAbs
# Scale each feature by its maximum absolute value.
# scaler = MaxAbsScaler()





scaler.fit(x_train) #이미 스케일러에 X 데이터의 max값 min값 저장돼 있음

x_train = scaler.transform(x_train)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)


x_test = scaler.transform(x_test)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)


x_predict = array([55, 65, 75]) # (3,)
x_predict = x_predict.reshape(1,3)
x_predict = scaler.transform(x_predict)
x_predict = x_predict.reshape(1,3,1)



# x_predict2 = scaler.transform(x_predict2)
# print(x)


#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM #LSTM도 layer


model = Sequential()
model.add(LSTM(30, activation='relu', input_shape=(3, 1)))
model.add(Dense(50, activation='relu')) #default activation = linear
model.add(Dense(70, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1)) #output: 1개



#3. 컴파일 및 훈련
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=80, mode='auto')

model.compile(loss='mse', optimizer='adam', metrics=['mse'])

model.fit(
    x_train,
    y_train,
    callbacks=[early_stopping],
    validation_split=0.2,
    epochs=1000, batch_size=10
)



#4. 평가, 예측

loss, mse = model.evaluate(x_test, y_test)
print("loss, mse: ", loss, mse)

y_predict = model.predict(x_predict)
print("예측값: ", y_predict)
