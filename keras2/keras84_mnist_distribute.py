#2020-12-08 ()
#distribute: GPU를 두 개 쓰겠다


import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist



(x_train, y_train), (x_test, y_test) = mnist.load_data() 


#1. 데이터 전처리: OneHotEncoding 대상은 Y
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train)

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.
                        #x_test.shape[0], x_test.shape[1] ... 



# #predict data, answer data
# x_predict = x_train[20:30]
# y_answer = y_train[20:30]


#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping


import tensorflow as tf
#parmeter 몇 개 알아 둘 것 
strategy = tf.distribute.MirroredStrategy(
    cross_device_ops=\
        tf.distribute.HierarchicalCopyAllReduce() # \ 넣으면 다음 줄까지 이어서 쓸 수 있음  

        
        # 장치 간 통신 방법을 바꾸고 싶다면, cross_device_ops 인자에 tf.distribute.CrossDeviceOps 타입의 인스턴스를 넘기면 됩니다. 
        # 현재 기본값인 tf.distribute.NcclAllReduce 이외에 
        # tf.distribute.HierarchicalCopyAllReduce와 tf.distribute.ReductionToOneDevice 두 가지 추가 옵션을 제공합니다.
        # 업데이트하면서 안 먹히는 옵션들이 생긴 듯... 지금 먹히는 건 얘 하나. 
        # 넣고 돌리면 GPU 두 개를 동시에 쓴다. 
        
) #분산처리할 준비가 strategey에 되어 있다 



with strategy.scope():

    model = Sequential()
    model.add(Conv2D(30, (2, 2), padding='same', input_shape=(28, 28, 1))) #padding 주의!
    model.add(Conv2D(50, (2, 2), padding='valid'))
    model.add(Conv2D(120, (3, 3))) #padding default=valid
    model.add(Conv2D(200, (2, 2), strides=2))
    model.add(Conv2D(30, (2, 2)))
    model.add(MaxPooling2D(pool_size=2)) #pool_size default=2
    model.add(Flatten()) 
    model.add(Dense(10, activation='relu')) 

    model.add(Dense(10, activation='softmax')) 


#3. 컴파일, 훈련

    early_stopping = EarlyStopping(monitor='loss', patience=30, mode='auto')

    model.compile(loss='categorical_crossentropy', 
                optimizer='adam', 
                metrics=['accuracy']) #"mean_squared_error" (풀네임도 가능하다)




model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=1,
          validation_split=0.2, callbacks=[early_stopping])



#4. 평가, 예측
#fit에서 쓴 이름과 맞춰 주기 

loss, accuracy = model.evaluate(x_test, y_test, batch_size=32)


print("=======mnist_CNN=======")
print("loss: ", loss)
print("acc: ", accuracy)


#y값 원상복구 
#np.argmax(, axis=1)


# #정답
# y_answer = np.argmax(y_answer, axis=1)

# #예측값
# y_predict = model.predict(x_predict)
# y_predict = np.argmax(y_predict, axis=1)


# print("예측값: ", y_predict)
# print("정답: ", y_answer)