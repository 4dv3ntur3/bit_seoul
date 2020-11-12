#1. 데이터
import numpy as np

x = np.array(range(1, 101))
y = np.array(range(101, 201))
# weight = 1, bias = 100

from sklearn.model_selection import train_test_split
# 잘려 나가는 순서 
x_train, x_test, y_train, y_test = train_test_split(
#   x, y, train_size=0.7)        
   x, y, test_size=0.3, shuffle=True)  # default: Shuffle = True, 섞고 싶지 않으면 False

# 모델 훈련시키는데 연속된 데이터로만 하면 weight=1이라고 과적화 됨 
# 항상 data 섞어 줘야 
print(x_test)


#2. 모델 구현
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()

model.add(Dense(300, #input_dim=1
         input_shape=(1,)))
model.add(Dense(50)) 
model.add(Dense(70))
model.add(Dense(90))
model.add(Dense(10))
model.add(Dense(70))
model.add(Dense(50))

model.add(Dense(1))


# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])

model.fit(x_train, y_train, batch_size=1, epochs=100, validation_split=0.2)


#4. 평가 예측
loss, acc = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE: ", RMSE(y_test, y_predict))


from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2: ", r2)
