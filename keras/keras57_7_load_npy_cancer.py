#2020-11-19 (9일차)
#*.npy로 저장한 dataset 불러오기: load_breast_cancer



#1. 데이터
import numpy as np
from numpy import array
from tensorflow.keras.utils import to_categorical

x = np.load('./data/cancer_x.npy')
y = np.load('./data/cancer_y.npy')


#전처리
from sklearn.model_selection import train_test_split 

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7
)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)



############## 1. load_model #############################

#3. 컴파일, 훈련

from tensorflow.keras.models import load_model
model1 = load_model('./save/cancer_dnn_model_weigths.h5')

#4. 평가, 예측
result1 = model1.evaluate(x_test, y_test, batch_size=32)
print("====model & weights 같이 저장=========")
print("loss : ", result1[0])
print("accuracy : ", result1[1])


############## 2. load_model ModelCheckPoint #############

model2 = load_model('./model/cancer_dnn_44-0.0016.hdf5')

#4. 평가, 예측

result2 = model2.evaluate(x_test, y_test, batch_size=32)
print("=======checkpoint 저장=========")
print("loss : ", result2[0])
print("accuracy : ", result2[1])


################ 3. load_weights ##################

#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout #LSTM도 layer


model3 = Sequential()
model3.add(Dense(64, activation='relu', input_shape=(x_train.shape[1],)))
model3.add(Dense(512, activation='relu'))
model3.add(Dropout(0.5))
model3.add(Dense(256, activation='relu'))
model3.add(Dropout(0.5))
model3.add(Dense(128, activation='relu'))
model3.add(Dense(300, activation='relu'))
model3.add(Dense(1024, activation='relu'))
model3.add(Dense(150, activation='relu'))
model3.add(Dense(70, activation='relu'))
model3.add(Dense(32, activation='relu'))
model3.add(Dense(1, activation='sigmoid')) #2진분류: sigmoid -> output: 0 or 1 이니까 1개임 


# 3. 컴파일
model3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model3.load_weights('./save/cancer_dnn_weights.h5')


#4. 평가, 예측
result3 = model3.evaluate(x_test, y_test, batch_size=32)
print("========weights 저장=========")
print("loss : ", result3[0])
print("accuracy : ", result3[1])



'''
====model & weights 같이 저장=========
loss :  0.007307616528123617
accuracy :  1.0
6/6 [==============================] - 0s 1ms/step - loss: 0.1741 - accuracy: 0.9708
=======checkpoint 저장=========
loss :  0.17407235503196716
accuracy :  0.9707602262496948
6/6 [==============================] - 0s 1ms/step - loss: 8.6444e-08 - acc: 1.0000
========weights 저장=========
loss :  8.64441744852229e-08
accuracy :  1.0
'''