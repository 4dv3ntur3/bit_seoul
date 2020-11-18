#2020-11-18 (8일차)
#MNIST -> CNN의 load_model


import numpy as np
from tensorflow.keras.datasets import mnist 

#1. 데이터 
(x_train, y_train), (x_test, y_test) = mnist.load_data()


#전처리: OneHotEncoding 대상은 Y
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.



from tensorflow.keras.models import load_model
##############1. load_model (fit 이후 save 모델) #############
#3. 컴파일, 훈련

model1 = load_model('./save/mnist_cnn_model_weights.h5')

#4. 평가, 예측
result1 = model1.evaluate(x_test, y_test, batch_size=32)
print("====model & weights 같이 저장=========")
print("loss : ", result1[0])
print("accuracy : ", result1[1])


############## 2. load_model ModelCheckPoint #############

model2 = load_model('./model/mnist-10-0.0694.hdf5')

#4. 평가, 예측

result2 = model2.evaluate(x_test, y_test, batch_size=32)
print("=======checkpoint 저장=========")
print("loss : ", result2[0])
print("accuracy : ", result2[1])


################ 3. load_weights ##################

# 2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

model3 = Sequential()
model3.add(Conv2D(30, (2, 2), padding='same', input_shape=(28, 28, 1))) #padding 주의!
model3.add(Conv2D(50, (2, 2), padding='valid'))
model3.add(Conv2D(120, (3, 3))) #padding default=valid
model3.add(Conv2D(200, (2, 2), strides=2))
model3.add(Conv2D(30, (2, 2)))
model3.add(MaxPooling2D(pool_size=2)) #pool_size default=2
model3.add(Flatten()) 
model3.add(Dense(10, activation='relu'))

model3.add(Dense(10, activation='softmax')) 


# 3. 컴파일
model3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model3.load_weights('./save/mnist_cnn_weights.h5')

#4. 평가, 예측
result3 = model3.evaluate(x_test, y_test, batch_size=32)
print("========weights 저장=========")
print("loss : ", result3[0])
print("accuracy : ", result3[1])



'''
====model & weights 같이 저장=========
loss :  0.0756116583943367
accuracy :  0.9807999730110168
313/313 [==============================] - 1s 2ms/step - loss: 0.0617 - accuracy: 0.9814
=======checkpoint 저장=========
loss :  0.061700958758592606
accuracy :  0.9814000129699707
313/313 [==============================] - 1s 2ms/step - loss: 0.0756 - acc: 0.9808
========weights 저장=========
loss :  0.0756116583943367
accuracy :  0.9807999730110168
'''
