#2020-11-18 (8일차)
#load_iris() -> CNN: checkpoints / model.fit() 이후 model.save() / model.save_weights()
#꽃잎과 줄기를 보고 어떤 꽃인지 판별하는 데이터 -> 다중분류
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
# modelpath = './model/iris-{epoch:02d}-{val_loss:.4f}.hdf5' 

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


early_stopping = EarlyStopping(monitor='loss', patience=10, mode='auto')

# cp = ModelCheckpoint(filepath=modelpath, 
#                      monitor='val_loss', 
#                      save_best_only=True, 
#                      mode='auto'
# )


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

hist = model.fit(
    x_train,
    y_train,
    callbacks=[early_stopping],
    validation_split=0.2,
    epochs=100, batch_size=32
)

#모델+가중치
model.save('./save/iris_cnn_model_weights.h5')

#가중치
model.save_weights('./save/iris_cnn_weights.h5')




#4. 평가, 예측
result = model.evaluate(x_test, y_test, batch_size=32)

# #fit에 있는 네 가지
# loss = hist.history['loss']
# val_loss = hist.history['val_loss']
# acc = hist.history['accuracy']
# val_acc = hist.history['val_accuracy']

# #시각화
# #plot에는 x, y가 들어간다 (그래야 그래프가 그려짐)
# import matplotlib.pyplot as plt

# plt.figure(figsize=(10, 6)) 
# #단위 뭔지 찾아볼 것!!!
# #pyplot.figure 는 매개 변수에 주어진 속성으로 새로운 도형을 생성합니다. 
# #figsize 는 도형 크기를 인치 단위로 정의합니다.


# plt.subplot(2, 1, 1) #2, 1, 1 -> 두 장 중의 첫 번째의 첫 번째 (2행 1열에서 첫 번째)
# # plt.plot(hist.history['loss'],) #loss값이 순서대로 감
# plt.plot(hist.history['loss'], marker='.', c='red', label='loss') #loss값이 순서대로 감, 마커는 점, 색깔은 빨간색, 라벨은 loss
# plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')

# plt.grid() #모눈종이 배경
# plt.title('loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')

# #위에서 라벨 명시 후 위치 명시
# #그림의 위치(location)는 상단: label:loss, label:val_loss 이 둘이 박스로 해서 저 위치에 나올 것
# plt.legend(loc='upper right')




# plt.subplot(2, 1, 2) #2, 1, 1 -> 2행 1열 중 두 번째 (두 번째 그림)
# # plt.plot(hist.history['loss'],) #loss값이 순서대로 감
# plt.plot(hist.history['accuracy'], marker='.', c='red') #loss값이 순서대로 감, 마커는 점, 색깔은 빨간색, 라벨은 loss
# plt.plot(hist.history['val_accuracy'], marker='.', c='blue')

# plt.grid() #모눈종이 배경
# plt.title('accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')

# #여긴 라벨만 명시
# plt.legend(['accuracy', 'val_accuracy'])

# #보여 줘
# plt.show()


print("=======iris_cnn=======")
model.summary()
print("loss: ", result[0])
print("acc: ", result[1])







#정답
# y_answer = np.argmax(y_answer, axis=1)

#예측값
# y_predict = model.predict(x_test)
# y_predict = np.argmax(y_predict, axis=1)

# print("예측값: ", y_predict)
# print("정답: ", y_test)


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
loss:  0.2875100076198578
acc:  0.9333333373069763



=======iris_dnn=======
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 64)                1984
_________________________________________________________________
dense_1 (Dense)              (None, 512)               33280
_________________________________________________________________
dropout (Dropout)            (None, 512)               0
_________________________________________________________________
dense_2 (Dense)              (None, 256)               131328
_________________________________________________________________
dropout_1 (Dropout)          (None, 256)               0
_________________________________________________________________
dense_3 (Dense)              (None, 128)               32896
_________________________________________________________________
dense_4 (Dense)              (None, 300)               38700
_________________________________________________________________
dense_5 (Dense)              (None, 150)               45150
_________________________________________________________________
dense_6 (Dense)              (None, 70)                10570
_________________________________________________________________
dense_7 (Dense)              (None, 32)                2272
_________________________________________________________________
dense_8 (Dense)              (None, 1)                 33
=================================================================
Total params: 296,213
Trainable params: 296,213
Non-trainable params: 0
_________________________________________________________________
loss:  0.013466596603393555
acc:  0.9941520690917969
예측값:  [1.1017855e-10 1.6328189e-13 1.3155242e-14 1.8938087e-11 9.9950278e-01
 1.6516525e-03 1.0000000e+00 9.9997473e-01 9.9999952e-01 9.9999988e-01]
정답:  [0 0 0 0 1 0 1 1 1 1]


'''
