#2020-11-18 (8일차)
#cifar-100 -> CNN: checkpoints / model.fit() 이후 model.save() / model.save_weights()
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



model.add(Conv2D(128, (2, 2), padding='valid', activation='relu', input_shape=(32, 32, 3)))
model.add(Conv2D(512, (2, 2), activation='relu'))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(Conv2D(128, (2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))

#output layer
model.add(Dense(100, activation='softmax')) #ouput 맞춰 줘야




#3. 컴파일, 훈련
#patience는 대개 10?

# modelpath = './model/cifar100_CNN-{epoch:02d}-{val_loss:.4f}.hdf5' 


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

early_stopping = EarlyStopping(monitor='loss', patience=10, mode='auto')

# cp = ModelCheckpoint(filepath=modelpath, 
#                      monitor='val_loss', 
#                      save_best_only=True, 
#                      mode='auto'
# ) 

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

hist = model.fit(x_train, y_train, epochs=100, batch_size=100, verbose=1,
          validation_split=0.3, callbacks=[early_stopping])

#fit에 있는 네 가지
loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']


#모델+가중치
model.save('./save/cifar100_cnn_model_weights.h5')
model.save_weights('./save/cifar100_cnn_weights.h5')



#4. 평가, 예측
result = model.evaluate(x_test, y_test, batch_size=100)

print("======cifar-100_CNN=======")
model.summary()


print("loss: ", result[0])
print("acc: ", result[1])

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


# #정답
# y_answer = np.argmax(y_answer, axis=1)

# #예측값
# y_predict = model.predict(x_predict)
# y_predict = np.argmax(y_predict, axis=1)

# print("예측값: ", y_predict)
# print("정답: ", y_answer)

model.summary()


'''
======cifar-100_CNN=======
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 31, 31, 128)       1664
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 30, 30, 512)       262656
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 28, 28, 256)       1179904
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 27, 27, 128)       131200
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 13, 13, 128)       0
_________________________________________________________________
flatten (Flatten)            (None, 21632)             0
_________________________________________________________________
dense (Dense)                (None, 1024)              22152192
_________________________________________________________________
dense_1 (Dense)              (None, 512)               524800
_________________________________________________________________
dropout (Dropout)            (None, 512)               0
_________________________________________________________________
dense_2 (Dense)              (None, 100)               51300
=================================================================
Total params: 24,303,716
Trainable params: 24,303,716
Non-trainable params: 0
_________________________________________________________________
loss:  10.299393653869629
acc:  0.2921999990940094
'''