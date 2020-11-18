#2020-11-18 (8일차)
#cifar-100 -> CNN: load_model
#1) 모델&가중치(fit 이후 save한 모델 / ModelCheckPoint / 가중치)



from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical
import numpy as np

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()


#전처리
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 3).astype('float32')/255.
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 3).astype('float32')/255.


from tensorflow.keras.models import load_model

##############1. load_model (fit 이후 save 모델) #############
#3. 컴파일, 훈련

model1 = load_model('./save/cifar100_cnn_model_weights.h5')

#4. 평가, 예측
result1 = model1.evaluate(x_test, y_test, batch_size=32)
print("====model & weights 같이 저장=========")
print("loss : ", result1[0])
print("accuracy : ", result1[1])


############## 2. load_model ModelCheckPoint #############

model2 = load_model('./model/cifar100_CNN-05-2.7898.hdf5')

#4. 평가, 예측

result2 = model2.evaluate(x_test, y_test, batch_size=32)
print("=======checkpoint 저장=========")
print("loss : ", result2[0])
print("accuracy : ", result2[1])


################ 3. load_weights ##################


#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensroflow.keras.layers import Dropout, Conv2D, MaxPooling2D, Flatten #LSTM도 layer

model3 = Sequential()
model3.add(Conv2D(128, (2, 2), padding='valid', activation='relu', input_shape=(32, 32, 3)))
model3.add(Conv2D(512, (2, 2), activation='relu'))
model3.add(Conv2D(256, (3, 3), activation='relu'))
model3.add(Conv2D(128, (2, 2), activation='relu'))
model3.add(MaxPooling2D(pool_size=2))
model3.add(Flatten())
model3.add(Dense(1024, activation='relu'))
model3.add(Dense(512, activation='relu'))
model3.add(Dropout(0.5))
model3.add(Dense(100, activation='softmax')) #ouput 맞춰 줘야



# 3. 컴파일
model3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model3.load_weights('./save/cifar100_cnn_weights.h5')


#4. 평가, 예측
result3 = model3.evaluate(x_test, y_test, batch_size=32)
print("========weights 저장=========")
print("loss : ", result3[0])
print("accuracy : ", result3[1])




'''
====model & weights 같이 저장=========
loss :  10.299388885498047
accuracy :  0.2921999990940094
313/313 [==============================] - 2s 7ms/step - loss: 2.7761 - accuracy: 0.3278
=======checkpoint 저장=========
loss :  2.7761313915252686
accuracy :  0.3278000056743622
313/313 [==============================] - 2s 7ms/step - loss: 10.2994 - acc: 0.2922
========weights 저장=========
loss :  10.299388885498047
accuracy :  0.2921999990940094
'''