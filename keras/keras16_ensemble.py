<<<<<<< HEAD
#2020-11-11 (3일차)
#앙상블(Ensemble): 모델 합치기
#X1, X2 -> Y1, Y2



#1. 데이터
import numpy as np

x1 = np.array([range(1, 101), range(711, 811), range(100)])
y1 = np.array([range(101, 201), range(311, 411), range(100)])

x1 = x1.T
y1 = y1.T


x2 = np.array([range(4, 104), range(761, 861), range(100)])
y2 = np.array([range(501, 601), range(431, 531), range(100, 200)])

x2 = x2.T
y2 = y2.T



#data 분리
from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(
    x1, y1, shuffle=True, train_size=0.7
)

x2_train, x2_test, y2_train, y2_test = train_test_split(
    x2, y2, shuffle=True, train_size=0.7
)



#2. 모델 구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

#두 모델은 각각 별개. 서로 영향을 끼치지 않는다.
#모델 1
input1 = Input(shape=(3,))
dense1_1 = Dense(10, activation='relu', name="dense1_1")(input1) 
dense1_2 = Dense(5, activation='relu', name="dense1_2")(dense1_1) 
dense1_3 = Dense(7, activation='relu', name="dense1_3")(dense1_2)
output1 = Dense(3)(dense1_3) #y도 100, 3임 
model1 = Model(inputs=input1, outputs=output1)

#model1.summary()

#모델 2
input2 = Input(shape=(3,))
dense2_1 = Dense(15, activation='relu', name="dense2_1")(input2) 
dense2_2 = Dense(11, activation='relu', name="dense2_2")(dense2_1) 
dense2_3 = Dense(3, activation='relu', name="dense2_3")(dense2_2)
output2 = Dense(3)(dense2_3) 
model2 = Model(inputs=input2, outputs=output2)

#model2.summary()


#모델 병합, concatenate
#성능 다 똑같지만, 대문자와 소문자는 사용 방법이 틀리다

from tensorflow.keras.layers import Concatenate, concatenate
# from keras.layers.merge import Concatenate, concatenate
# from keras.layers import Concatenate, concatenate

#2개 이상 합칠 땐 list 사용*****
#밑에서 합쳐진다 
#함수형 모델
#middle1 = Dense(7)(middle1)
#middle1 = Dense(11)(middle1)
#이래도 된다! 

#merge1 = concatenate([output1, output2])
#class니까 생성자를
#Concatenate(axis=1)
merge1 = Concatenate()([output1, output2])
middle1 = Dense(30)(merge1)
middle1 = Dense(7)(middle1)
middle1 = Dense(11)(middle1)



################ ouput 모델 구성 (분기)
output1 = Dense(30)(middle1)
output1 = Dense(7)(output1)
output1 = Dense(3)(output1)

output2 = Dense(15)(middle1)
output2_1 = Dense(14)(output2)
output2_3 = Dense(11)(output2_1)
output2_4 = Dense(3)(output2_3)


#모델 정의
model = Model(inputs=[input1, input2], outputs=[output1, output2_4])

model.summary()

#concatenate는 연산하지 않는다 

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics='mse')
model.fit(
    [x1_train, x2_train],
    [y1_train, y2_train],
    epochs=100, batch_size=8,
    validation_split=0.25,
    verbose=1
)


#4. 평가
result = model.evaluate([x1_test, x2_test], [y1_test, y2_test], 
                batch_size=8)

#수치가 5개인 이유: 전체 loss, 각각 두 아웃풋들의 loss 및 mse
#전체 loss = 각 아웃풋의 loss+mse 합 




#RMSE, R2
y_pred_1, y_pred_2 = model.predict([x1_test, x2_test])

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
    # predict해서 나온 값과 원래 y_test 값을 비교해서 RMSE로 나오게 하겠다

print("\nRMSE_1: ", RMSE(y1_test, y_pred_1))
print("RMSE_2: ", RMSE(y2_test, y_pred_2), "\n")


# R2는 함수 제공
from sklearn.metrics import r2_score
print("R2_1: ", r2_score(y1_test, y_pred_1))
=======
#2020-11-11 (3일차)
#앙상블(Ensemble): 모델 합치기
#X1, X2 -> Y1, Y2



#1. 데이터
import numpy as np

x1 = np.array([range(1, 101), range(711, 811), range(100)])
y1 = np.array([range(101, 201), range(311, 411), range(100)])

x1 = x1.T
y1 = y1.T


x2 = np.array([range(4, 104), range(761, 861), range(100)])
y2 = np.array([range(501, 601), range(431, 531), range(100, 200)])

x2 = x2.T
y2 = y2.T



#data 분리
from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(
    x1, y1, shuffle=True, train_size=0.7
)

x2_train, x2_test, y2_train, y2_test = train_test_split(
    x2, y2, shuffle=True, train_size=0.7
)



#2. 모델 구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

#두 모델은 각각 별개. 서로 영향을 끼치지 않는다.
#모델 1
input1 = Input(shape=(3,))
dense1_1 = Dense(10, activation='relu', name="dense1_1")(input1) 
dense1_2 = Dense(5, activation='relu', name="dense1_2")(dense1_1) 
dense1_3 = Dense(7, activation='relu', name="dense1_3")(dense1_2)
output1 = Dense(3)(dense1_3) #y도 100, 3임 
model1 = Model(inputs=input1, outputs=output1)

#model1.summary()

#모델 2
input2 = Input(shape=(3,))
dense2_1 = Dense(15, activation='relu', name="dense2_1")(input2) 
dense2_2 = Dense(11, activation='relu', name="dense2_2")(dense2_1) 
dense2_3 = Dense(3, activation='relu', name="dense2_3")(dense2_2)
output2 = Dense(3)(dense2_3) 
model2 = Model(inputs=input2, outputs=output2)

#model2.summary()


#모델 병합, concatenate
#성능 다 똑같지만, 대문자와 소문자는 사용 방법이 틀리다

from tensorflow.keras.layers import Concatenate, concatenate
# from keras.layers.merge import Concatenate, concatenate
# from keras.layers import Concatenate, concatenate

#2개 이상 합칠 땐 list 사용*****
#밑에서 합쳐진다 
#함수형 모델
#middle1 = Dense(7)(middle1)
#middle1 = Dense(11)(middle1)
#이래도 된다! 

#merge1 = concatenate([output1, output2])
#class니까 생성자를
#Concatenate(axis=1)
merge1 = Concatenate()([output1, output2])
middle1 = Dense(30)(merge1)
middle1 = Dense(7)(middle1)
middle1 = Dense(11)(middle1)



################ ouput 모델 구성 (분기)
output1 = Dense(30)(middle1)
output1 = Dense(7)(output1)
output1 = Dense(3)(output1)

output2 = Dense(15)(middle1)
output2_1 = Dense(14)(output2)
output2_3 = Dense(11)(output2_1)
output2_4 = Dense(3)(output2_3)


#모델 정의
model = Model(inputs=[input1, input2], outputs=[output1, output2_4])

model.summary()

#concatenate는 연산하지 않는다 

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics='mse')
model.fit(
    [x1_train, x2_train],
    [y1_train, y2_train],
    epochs=100, batch_size=8,
    validation_split=0.25,
    verbose=1
)


#4. 평가
result = model.evaluate([x1_test, x2_test], [y1_test, y2_test], 
                batch_size=8)

#수치가 5개인 이유: 전체 loss, 각각 두 아웃풋들의 loss 및 mse
#전체 loss = 각 아웃풋의 loss+mse 합 




#RMSE, R2
y_pred_1, y_pred_2 = model.predict([x1_test, x2_test])

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
    # predict해서 나온 값과 원래 y_test 값을 비교해서 RMSE로 나오게 하겠다

print("\nRMSE_1: ", RMSE(y1_test, y_pred_1))
print("RMSE_2: ", RMSE(y2_test, y_pred_2), "\n")


# R2는 함수 제공
from sklearn.metrics import r2_score
print("R2_1: ", r2_score(y1_test, y_pred_1))
>>>>>>> b4eac5d0e44c2b94dcc0999e9053d1954a85c531
print("R2_2: ", r2_score(y2_test, y_pred_2), "\n")