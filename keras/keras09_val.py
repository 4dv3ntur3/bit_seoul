#2020-11-10 (2일차)
#검증(validation) 데이터 추가, fit에서 validation_data=( , )


import numpy as np

#1. 데이터
# 정제된 1부터 20까지의 데이터를 1~10까진 train, 11~15 validation, // 여기까진 훈련에 들어감
# 16~20을 test

x_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

x_val = np.array([11, 12, 13, 14, 15])
y_val = np.array([11, 12, 13, 14, 15])

# x_pred = np.array([16, 17, 18])

x_test = np.array([16, 17, 18, 19, 20])
y_test = np.array([16, 17, 18, 19, 20])


from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense


#2. 모델 구성 
model = Sequential() 

model.add(Dense(300, input_dim=1)) # input_dim & 마지막 dense 동일해야

# hidden
model.add(Dense(50)) 
model.add(Dense(70))
model.add(Dense(90))
model.add(Dense(10))
model.add(Dense(70))
model.add(Dense(50))
# layers

model.add(Dense(1))


#3. 컴파일, 훈련 
model.compile(loss='mse', optimizer='adam', metrics=['mae']) 

#model.fit(x, y, epochs=100, batch_size=1)
model.fit(x_train, y_train, epochs=1000,
                            validation_data=(x_val, y_val)) 

#4. 평가, 예측
# loss, acc = model.evaluate(x, y, bacth_size=1)
# loss, acc = model.evaluate(x, y)
loss = model.evaluate(x_test, y_test)

print("loss : ", loss)
# print("acc : ", acc)

# 원래는 새 값에 대한 새 예측을 찾는 용도지만
# predict로 값을 만들고 원래 나와야 하는 정답(y_test)와 비교하는 것
y_predict = model.predict(x_test)
print("결과물 : \n : ", y_predict)



# 사이킷런 활용

# R2와 RMSE를 보통 함께 지표로 활용한다 (보완지표)
# RMSE
from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
    # predict해서 나온 값과 원래 y_test 값을 비교해서 RMSE로 나오게 하겠다
print("RMSE: ", RMSE(y_test, y_predict))

# R2는 함수 제공
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2: ", r2)
