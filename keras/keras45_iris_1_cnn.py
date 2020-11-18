#2020-11-18 (8일차)
#다중분류
#꽃잎과 줄기를 보고 어떤 꽃인지 판별하는 데이터
#x column=4 y label:1

import numpy as np
from sklearn.datasets import load_iris

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, LSTM
from tensorflow.keras.layers import Flatten, MaxPooling2D, Dropout



#1. 데이터 

#데이터 구조 확인

dataset = load_iris() #data(X)와 target(Y)으로 구분되어 있다
x = dataset.data
y = dataset.target


print(x.shape) #(150, 4)
print(y.shape) #(150,)




#전처리
from sklearn.model_selection import train_test_split 

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, shuffle=True
)

x_train = x_train.reshape(x_train.shape[0], 4, 1, 1).astype('float32')/255.
x_test = x_test.reshape(x_test.shape[0], 4, 1, 1).astype('float32')/255.

#OneHotEncoding (다중분류)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape)

x_predict = x_train[:10]
y_answer = y_train[:10]


#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv2D, MaxPooling2D, Flatten #LSTM도 layer


model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(4, 1, 1))) #padding 주의!
# model.add(Conv2D(32, (3, 3), padding='same',activation='relu'))
# model.add(MaxPooling2D(pool_size=2)) #pool_size default=2
model.add(Dropout(0.2))


model.add(Conv2D(64, (3, 3), padding='same', activation='relu')) #padding default=valid
# model.add(Conv2D(128, (3, 3), padding='same',activation='relu'))
# model.add(MaxPooling2D(pool_size=2)) #pool_size default=2
model.add(Dropout(0.2))

model.add(Conv2D(256, (3, 3), padding='same', activation='relu')) #padding default=valid
# model.add(Conv2D(32, (3, 3), padding='same',activation='relu'))
# model.add(MaxPooling2D(pool_size=2)) #pool_size default=2
model.add(Dropout(0.2))


model.add(Flatten())
model.add(Dense(1024, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax')) #ouput 





#3. 컴파일 및 훈련
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=85, mode='auto')

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(
    x_train,
    y_train,
    callbacks=[early_stopping],
    validation_split=0.2,
    epochs=1000, batch_size=32
)



#4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test, batch_size=32)

print("=======iris_cnn=======")
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
=======iris_cnn=======
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 4, 1, 32)          320
_________________________________________________________________
dropout (Dropout)            (None, 4, 1, 32)          0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 4, 1, 64)          18496
_________________________________________________________________
dropout_1 (Dropout)          (None, 4, 1, 64)          0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 4, 1, 256)         147712
_________________________________________________________________
dropout_2 (Dropout)          (None, 4, 1, 256)         0
_________________________________________________________________
flatten (Flatten)            (None, 1024)              0
_________________________________________________________________
dense (Dense)                (None, 1024)              1049600
_________________________________________________________________
dense_1 (Dense)              (None, 3)                 3075
=================================================================
Total params: 1,219,203
Trainable params: 1,219,203
Non-trainable params: 0
_________________________________________________________________
loss:  0.057593587785959244
acc:  0.9777777791023254
예측값:  [0 2 2 2 2 2 2 0 0 1]
정답:  [0 2 2 2 2 2 2 0 0 1]
PS D:\Study>
'''