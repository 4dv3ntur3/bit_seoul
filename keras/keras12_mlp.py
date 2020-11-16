<<<<<<< HEAD
#2020-11-10 (2일차)
#Multi Layer Perceptron
#스칼라, 벡터, 행렬, 텐서 
#shape 맞추기****

#1. 데이터
import numpy as np

x = np.array([range(1, 101), range(711, 811), range(100)])
y = np.array([range(101, 201), range(311, 411), range(100)])


'''
print(x.shape) # (3, 100)

# 과제: (100, 3)으로 변환해야 한다

1) numpy의 transpose() 이용
x = np.transpose(x)
y = np.transpose(y)

2) numpy array 객체의 T attribute
x = x.T
y = y.T

3) transpose
x = x.transpose()
y = y.transpose()
'''

x = x.transpose()
y = y.transpose()


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7)



# slicing
# x_train = x[:60]
# y_train = y[:60]
# x_test = x[60:80]
# y_test = y[60:80]


'''
#확인용

#for i in range(0, 50):
# print(i+1, ". ", "x: ", x[i], "\n", "     y: ", y[i])

#print(x)
#print(y)

#print(x.shape)
#print(y.shape)
'''

#행 무시, 열 우선
#data의 특성은 "열"에 의해 결정된다
#특성=feature=column=열


#최종목표: y1, y2, y3 = w1x1 + w2x2 + w3x3 + b (y도 3개)

#2. 모델 구성
from tensorflow.keras.models import Sequential #순차적 모델
from tensorflow.keras.layers import Dense #DNN 구조의 Dense모델

model = Sequential()
model.add(Dense(100, input_dim=3)) #column개수=3개
model.add(Dense(5))
model.add(Dense(7))
model.add(Dense(9))
model.add(Dense(3)) # 출력 =3개



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=100, validation_split=0.2)

#4. 평가 예측
loss, acc = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE: ", RMSE(y_test, y_predict))


from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
=======
#2020-11-10 (2일차)
#Multi Layer Perceptron
#스칼라, 벡터, 행렬, 텐서 
#shape 맞추기****

#1. 데이터
import numpy as np

x = np.array([range(1, 101), range(711, 811), range(100)])
y = np.array([range(101, 201), range(311, 411), range(100)])


'''
print(x.shape) # (3, 100)

# 과제: (100, 3)으로 변환해야 한다

1) numpy의 transpose() 이용
x = np.transpose(x)
y = np.transpose(y)

2) numpy array 객체의 T attribute
x = x.T
y = y.T

3) transpose
x = x.transpose()
y = y.transpose()
'''

x = x.transpose()
y = y.transpose()


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7)



# slicing
# x_train = x[:60]
# y_train = y[:60]
# x_test = x[60:80]
# y_test = y[60:80]


'''
#확인용

#for i in range(0, 50):
# print(i+1, ". ", "x: ", x[i], "\n", "     y: ", y[i])

#print(x)
#print(y)

#print(x.shape)
#print(y.shape)
'''

#행 무시, 열 우선
#data의 특성은 "열"에 의해 결정된다
#특성=feature=column=열


#최종목표: y1, y2, y3 = w1x1 + w2x2 + w3x3 + b (y도 3개)

#2. 모델 구성
from tensorflow.keras.models import Sequential #순차적 모델
from tensorflow.keras.layers import Dense #DNN 구조의 Dense모델

model = Sequential()
model.add(Dense(100, input_dim=3)) #column개수=3개
model.add(Dense(5))
model.add(Dense(7))
model.add(Dense(9))
model.add(Dense(3)) # 출력 =3개



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=100, validation_split=0.2)

#4. 평가 예측
loss, acc = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE: ", RMSE(y_test, y_predict))


from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
>>>>>>> b4eac5d0e44c2b94dcc0999e9053d1954a85c531
print("R2: ", r2)