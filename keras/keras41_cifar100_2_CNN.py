#2020-11-17 (7일차)
#cifar-100 -> CNN
#color이므로 channel=3

from tensorflow.keras.datasets import cifar100

#이미지 분류-> OneHotEncoding
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, LSTM, Dropout
from tensorflow.keras.layers import Flatten, MaxPooling2D #maxpooling2d는 들어가도 되고 안 들어가도 됨 필수 아님
import matplotlib.pyplot as plt
import numpy as np


(x_train, y_train), (x_test, y_test) = cifar100.load_data()

#데이터 구조 확인
# print(x_train.shape, x_test.shape) #(50000, 32, 32, 3) (10000, 32, 32, 3)
# print(y_train.shape, y_test.shape) #(50000, 1) (10000, 1)


#1. 데이터 전처리
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 3).astype('float32')/255.
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 3).astype('float32')/255.


x_predict = x_train[:10]
y_answer = y_train[:10]


#2. 모델
model = Sequential()

#input layer


#hidden layer
# model.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3))) #padding 주의!
# model.add(Dropout(0.3))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=2)) #pool_size default=2


# model.add(Conv2D(128, (3, 3), padding='same', activation='relu')) #padding default=valid
# model.add(Dropout(0.4))
# model.add(Conv2D(128, (3, 3), activation='relu', padding='same',))
# model.add(MaxPooling2D(pool_size=2)) #pool_size default=2


# model.add(Conv2D(256, (3, 3), padding='same', activation='relu')) #padding default=valid
# model.add(Dropout(0.4))

# model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
# # model.add(MaxPooling2D(pool_size=2)) #pool_size default=2
# model.add(Dropout(0.4))

# model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
# model.add(MaxPooling2D(pool_size=2)) #pool_size default=2


# model.add(Conv2D(512, (3, 3), padding='same', activation='relu')) #padding default=valid
# model.add(Dropout(0.4))

# model.add(Conv2D(512, (3, 3), padding='same', activation='relu')) #padding default=valid
# model.add(Dropout(0.4))

# model.add(Conv2D(512, (3, 3), padding='same', activation='relu')) #padding default=valid
# model.add(MaxPooling2D(pool_size=2)) #pool_size default=2

# model.add(Conv2D(512, (3, 3), padding='same', activation='relu')) #padding default=valid
# model.add(Dropout(0.4))

# model.add(Conv2D(512, (3, 3), padding='same', activation='relu')) #padding default=valid
# model.add(Dropout(0.5))



# model.add(Flatten())
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.5))

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3))) #padding 주의!
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=2)) #pool_size default=2

model.add(Conv2D(64, (3, 3), padding='same', activation='relu')) #padding default=valid
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=2)) #pool_size default=2
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), padding='same', activation='relu')) #padding default=valid
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=2)) #pool_size default=2
model.add(Dropout(0.3))

# model.add(Conv2D(256, (3, 3), padding='same', activation='relu')) #padding default=valid
# model.add(Conv2D(256, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=2)) #pool_size default=2
# model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='softmax')) #cifar-100은 class가 100개




#3. 컴파일, 훈련
#patience는 대개 10% 정도
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='loss', patience=10, mode='auto')

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=100, batch_size=512, verbose=1,
          validation_split=0.2, callbacks=[early_stopping])



#4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test, batch_size=512)

print("======cifar-100_CNN=======")
model.summary()


print("loss: ", loss)
print("acc: ", accuracy)


#정답
y_answer = np.argmax(y_answer, axis=1)

#예측값
y_predict = model.predict(x_predict)
y_predict = np.argmax(y_predict, axis=1)

print("예측값: ", y_predict)
print("정답: ", y_answer)




'''
======cifar-100_CNN=======
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 32, 32, 32)        896
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 30, 30, 32)        9248
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 15, 15, 32)        0
_________________________________________________________________
dropout (Dropout)            (None, 15, 15, 32)        0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 15, 15, 64)        18496
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 13, 13, 64)        36928
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 6, 6, 64)          0
_________________________________________________________________
dropout_1 (Dropout)          (None, 6, 6, 64)          0
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 6, 6, 128)         73856
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 4, 4, 128)         147584
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 2, 2, 128)         0
_________________________________________________________________
dropout_2 (Dropout)          (None, 2, 2, 128)         0
_________________________________________________________________
flatten (Flatten)            (None, 512)               0
_________________________________________________________________
dense (Dense)                (None, 1024)              525312
_________________________________________________________________
dropout_3 (Dropout)          (None, 1024)              0
_________________________________________________________________
dense_1 (Dense)              (None, 100)               102500
=================================================================
Total params: 914,820
Trainable params: 914,820
Non-trainable params: 0
_________________________________________________________________
loss:  2.1102476119995117
acc:  0.48500001430511475
예측값:  [19 29  0 11  1 86 90 28 23 31]
정답:  [19 29  0 11  1 86 90 28 23 31]
'''