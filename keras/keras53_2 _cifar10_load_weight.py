#2020-11-18 (8일차)
#Cifar10 -> CNN의 load_model
#1) 모델&가중치(fit 이후 save한 모델 / ModelCheckPoint / 가중치)



import numpy as np
from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data() #괄호 주의


#1. 데이터 전처리: OneHotEncoding 대상은 Y
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train)


x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 3).astype('float32')/255.
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 3).astype('float32')/255.



from tensorflow.keras.models import load_model
############# 1. load_model (fit 이후 save 모델) ##############
#3. 컴파일, 훈련

model1 = load_model('./save/cifar10_cnn_model.h5')

#4. 평가, 예측
result1 = model1.evaluate(x_test, y_test, batch_size=32)
print("====model & weights 같이 저장=========")
print("loss : ", result1[0])
print("accuracy : ", result1[1])


############## 2. load_model ModelCheckPoint #############

model2 = load_model('./model/cifar-10_CNN-61-0.6488.hdf5')

#4. 평가, 예측

result2 = model2.evaluate(x_test, y_test, batch_size=32)
print("=======checkpoint 저장=========")
print("loss : ", result2[0])
print("accuracy : ", result2[1])


################ 3. load_weights ##################

#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

model3 = Sequential()
model3.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3))) #padding 주의!
model3.add(Conv2D(32, (3, 3), activation='relu'))
model3.add(MaxPooling2D(pool_size=2)) #pool_size default=2
model3.add(Dropout(0.2))


model3.add(Conv2D(64, (3, 3), padding='same', activation='relu')) #padding default=valid
model3.add(Conv2D(64, (3, 3), activation='relu'))
model3.add(MaxPooling2D(pool_size=2)) #pool_size default=2
model3.add(Dropout(0.2))

model3.add(Conv2D(128, (3, 3), padding='same', activation='relu')) #padding default=valid
model3.add(Conv2D(128, (3, 3), activation='relu'))
model3.add(MaxPooling2D(pool_size=2)) #pool_size default=2
model3.add(Dropout(0.2))


model3.add(Flatten())
model3.add(Dense(1024, activation='relu'))
model3.add(Dropout(0.5))
model3.add(Dense(10, activation='softmax')) #ouput 


# 3. 컴파일

model3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model3.load_weights('./save/cifar10_cnn_weights.h5')


#4. 평가, 예측
result3 = model3.evaluate(x_test, y_test, batch_size=32)
print("========weights 저장=========")
print("loss : ", result3[0])
print("accuracy : ", result3[1])





'''
====model & weights 같이 저장=========
loss :  0.7228475213050842
accuracy :  0.7699000239372253
313/313 [==============================] - 1s 2ms/step - loss: 0.6565 - accuracy: 0.7829
=======checkpoint 저장=========
loss :  0.656539797782898
accuracy :  0.7828999757766724
313/313 [==============================] - 1s 2ms/step - loss: 0.7228 - acc: 0.7699
========weights 저장=========
loss :  0.7228475213050842
accuracy :  0.7699000239372253
'''
