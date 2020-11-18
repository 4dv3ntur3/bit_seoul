#2020-11-18 (8일차)
#load_iris -> CNN: load_model
#1) 모델&가중치(fit 이후 save한 모델 / ModelCheckPoint / 가중치)


import numpy as np
from sklearn.datasets import load_iris

from tensorflow.keras.utils import to_categorical


#1. 데이터 

#데이터 구조 확인
dataset = load_iris() #data(X)와 target(Y)으로 구분되어 있다

x = dataset.data #(150, 4)
y = dataset.target #(150,)



#전처리
from sklearn.model_selection import train_test_split 

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, shuffle=True
)

x_train = x_train.reshape(x_train.shape[0], 4, 1, 1).astype('float32')/255.
x_test = x_test.reshape(x_test.shape[0], 4, 1, 1).astype('float32')/255.

#OneHotEncoding (다중 분류)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)





############## 1. load_model (fit 이후 save 모델) #############
#3. 컴파일, 훈련

from tensorflow.keras.models import load_model
model1 = load_model('./save/iris_cnn_model_weights.h5')

#4. 평가, 예측
result1 = model1.evaluate(x_test, y_test, batch_size=32)
print("====model & weights 같이 저장=========")
print("loss : ", result1[0])
print("accuracy : ", result1[1])


############## 2. load_model ModelCheckPoint #############

model2 = load_model('./model/iris-432-0.0332.hdf5')

#4. 평가, 예측

result2 = model2.evaluate(x_test, y_test, batch_size=32)
print("=======checkpoint 저장=========")
print("loss : ", result2[0])
print("accuracy : ", result2[1])


################ 3. load_weights ##################

#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv2D, MaxPooling2D, Flatten #LSTM도 layer


model3 = Sequential()
model3.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(4, 1, 1))) 
# model.add(Conv2D(32, (3, 3), padding='same',activation='relu'))
# model.add(MaxPooling2D(pool_size=2)) #pool_size default=2
model3.add(Dropout(0.2))


model3.add(Conv2D(64, (3, 3), padding='same', activation='relu')) #padding default=valid
# model.add(Conv2D(128, (3, 3), padding='same',activation='relu'))
# model.add(MaxPooling2D(pool_size=2)) #pool_size default=2
model3.add(Dropout(0.2))

model3.add(Conv2D(256, (3, 3), padding='same', activation='relu')) #padding default=valid
# model.add(Conv2D(32, (3, 3), padding='same',activation='relu'))
# model.add(MaxPooling2D(pool_size=2)) #pool_size default=2
model3.add(Dropout(0.2))


model3.add(Flatten())
model3.add(Dense(1024, activation='relu'))
# model.add(Dropout(0.2))
model3.add(Dense(3, activation='softmax')) #ouput 



# 3. 컴파일
model3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model3.load_weights('./save/iris_cnn_weights.h5')


#4. 평가, 예측
result3 = model3.evaluate(x_test, y_test, batch_size=32)
print("========weights 저장=========")
print("loss : ", result3[0])
print("accuracy : ", result3[1])




'''
====model & weights 같이 저장=========
loss :  0.30430975556373596
accuracy :  0.9333333373069763
2/2 [==============================] - 0s 2ms/step - loss: 0.0348 - accuracy: 1.0000
=======checkpoint 저장=========
loss :  0.03483808413147926
accuracy :  1.0
2/2 [==============================] - 0s 2ms/step - loss: 0.3043 - acc: 0.9333
========weights 저장=========
loss :  0.30430975556373596
accuracy :  0.9333333373069763
'''