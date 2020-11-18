#2020-11-18 (8일차)
#MNIST -> CNN의 load_model, load_weights, load_checkpoint


import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist #텐서플로우에서 제공해 준다(수치로 변환해서 제공)

#train_test_split 할 필요 없이 알아서 나눠 준다
(x_train, y_train), (x_test, y_test) = mnist.load_data() #괄호 주의

#60000장 * 28pixel * 28pixel
# print(x_train.shape, x_test.shape) #(60000, 28, 28)(10000, 28, 28)
# print(y_train.shape, y_test.shape) #(60000, )      (10000,)        : 스칼라


# print(x_train[0])
# print(y_train[1]) #label 



# plt.imshow(x_train[0], 'gray')
# plt.show()


#8은 2보다 4배의 가치? 3은 1보다 3배의 가치? no
#One-Hot Encoder
#y_train: 60000, -> OneHotEncoding : 1 0 0 0 0 0 0 0 0 0 (60000, 10) : 분류가 10개니까 (0~9)



#1. 데이터 전처리: OneHotEncoding 대상은 Y
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train)


# print(y_train.shape, y_test.shape)
# print(y_train[0]) #y_train[0]=5 -> [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]


#shape 바꿀 줄 알아야 함
#60000, 14, 14, 4도 가능하고 60000, 28, 14, 2도 가능
#LSTM으로도 바꿀 수 있다

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.
                        #x_test.shape[0], x_test.shape[1] ... 



#predict data, answer data
x_predict = x_train[20:30]
y_answer = y_train[20:30]



#CNN에 넣을 수 있는 4차원 reshape + y도 onehotencoding
#scaler 사용해야: 어떤 게 더 좋을지는 해 봐야 안다
#지금 이 상황에서 M은 255라는 걸 알고 있음. 그러므로 MinMax에서는 255로 나누면 0~1 사이로 수렴 가능


# print(x_train[0]) 


#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

model = Sequential()
model.add(Conv2D(30, (2, 2), padding='same', input_shape=(28, 28, 1))) #padding 주의!
model.add(Conv2D(50, (2, 2), padding='valid'))
model.add(Conv2D(120, (3, 3))) #padding default=valid
model.add(Conv2D(200, (2, 2), strides=2))
model.add(Conv2D(30, (2, 2)))
model.add(MaxPooling2D(pool_size=2)) #pool_size default=2
model.add(Flatten()) 
model.add(Dense(10, activation='relu')) #flatten하면서 곱하고 dense에서 또 100 곱함 
                                        #Conv2d의 activation default='relu'
                                        #LSTM의 activation default='tanh'
#MaxPooling2D-Flatten:reshape 개념

model.add(Dense(10, activation='softmax')) #softmax** : 2 이상 분류(다중분류)의 activation은 softmax, 2진분류는 sigmoid
                                            #즉 softmax를 사용하려면 OneHotEncoding 해야




#========= 1. load_model (fit 이후 save 모델) ===================
#3. 컴파일, 훈련

from tensorflow.keras.models import load_model
model1 = load_model('./save/mnist_cnn_model_weights.h5')

#4. 평가, 예측
result1 = model1.evaluate(x_test, y_test, batch_size=32)
print("====model & weights 같이 저장=========")
print("loss : ", result1[0])
print("accuracy : ", result1[1])


############## 2. load_model ModelCheckPoint #############

from tensorflow.keras.models import load_model
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
model3.add(Dense(10, activation='relu')) #flatten하면서 곱하고 dense에서 또 100 곱함 
                                        #Conv2d의 activation default='relu'
                                        #LSTM의 activation default='tanh'
#MaxPooling2D-Flatten:reshape 개념

model3.add(Dense(10, activation='softmax')) #softmax** : 2 이상 분류(다중분류)의 activation은 softmax, 2진분류는 sigmoid
                                            #즉 softmax를 사용하려면 OneHotEncoding 해야


# model.summary()

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
