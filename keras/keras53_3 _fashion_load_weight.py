#2020-11-18 (8일차)
#fashon_mnist -> CNN + ModelCheckPoint
#흑백


from tensorflow.keras.datasets import fashion_mnist

#이미지 분류-> OneHotEncoding
from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt
import numpy as np


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# print(x_train[0])
# print("y_train[0]: ", y_train[0])

#데이터 구조 확인
print(x_train.shape, x_test.shape) #(60000, 28, 28) (10000, 28, 28)        
print(y_train.shape, y_test.shape) #(60000,) (10000,)


#1. 데이터 전처리
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# print(y_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1).astype('float32')/255.
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1).astype('float32')/255.


# x_predict = x_train[:10]
# y_answer = y_train[:10]





#========= 1. load_model (fit 이후 save 모델) ===================
#3. 컴파일, 훈련

from tensorflow.keras.models import load_model
model1 = load_model('./save/fashion_cnn_model_weights.h5')

#4. 평가, 예측
result1 = model1.evaluate(x_test, y_test, batch_size=32)
print("====model & weights 같이 저장=========")
print("loss : ", result1[0])
print("accuracy : ", result1[1])


############## 2. load_model ModelCheckPoint #############

from tensorflow.keras.models import load_model
model2 = load_model('./model/fashion-74-0.3656.hdf5')

#4. 평가, 예측

result2 = model2.evaluate(x_test, y_test, batch_size=32)
print("=======checkpoint 저장=========")
print("loss : ", result2[0])
print("accuracy : ", result2[1])


################ 3. load_weights ##################

#2. 모델
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, LSTM
from tensorflow.keras.layers import Flatten, MaxPooling2D #maxpooling2d는 들어가도 되고 안 들어가도 됨 필수 아님
model3 = Sequential()
model3.add(Conv2D(200, (3, 3), padding='same', input_shape=(x_train.shape[1], x_train.shape[2], 1))) #padding 주의!
model3.add(Conv2D(180, (2, 2), padding='valid'))
model3.add(Conv2D(100, (3, 3), strides=2)) #padding default=valid
model3.add(Conv2D(50, (2, 2)))
model3.add(Conv2D(30, (2, 2)))
model3.add(Conv2D(10, (3, 3)))
model3.add(MaxPooling2D(pool_size=2)) #pool_size default=2
model3.add(Flatten()) 
model3.add(Dense(10, activation='relu')) #flatten하면서 곱하고 dense에서 또 100 곱함 
                                        #Conv2d의 activation default='relu'
                                        #LSTM의 activation default='tanh'
# model.add(Dense(50, activation='relu'))
model3.add(Dense(10, activation='softmax')) #label: 0~9 (항상 dataset label 확인)



# model.summary()

# 3. 컴파일

model3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model3.load_weights('./save/fashion_cnn_weights.h5')


#4. 평가, 예측
result3 = model3.evaluate(x_test, y_test, batch_size=32)
print("========weights 저장=========")
print("loss : ", result3[0])
print("accuracy : ", result3[1])



'''
====model & weights 같이 저장=========
loss :  0.3941965699195862
accuracy :  0.8639000058174133
313/313 [==============================] - 1s 3ms/step - loss: 0.3869 - accuracy: 0.8676
=======checkpoint 저장=========
loss :  0.38687172532081604
accuracy :  0.8676000237464905
313/313 [==============================] - 1s 3ms/step - loss: 0.3942 - acc: 0.8639
========weights 저장=========
loss :  0.3941965699195862
accuracy :  0.8639000058174133
'''