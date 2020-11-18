#2020-11-18 (8일차)
#cifar-10 -> CNN: checkpoints / model.fit() 이후 model.save() / model.save_weights()

from tensorflow.keras.datasets import cifar10

#이미지 분류-> OneHotEncoding
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, LSTM
from tensorflow.keras.layers import Flatten, MaxPooling2D, Dropout #maxpooling2d는 들어가도 되고 안 들어가도 됨 필수 아님
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#데이터 구조 확인
# print(x_train.shape, x_test.shape) #(50000, 32, 32, 3) (10000, 32, 32, 3)
# print(y_train.shape, y_test.shape) #(50000, 1) (10000, 1)


#1. 데이터 전처리
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 3).astype('float32')/255.
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 3).astype('float32')/255.




#2. 모델
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3))) #padding 주의!
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=2)) #pool_size default=2
model.add(Dropout(0.2))


model.add(Conv2D(64, (3, 3), padding='same', activation='relu')) #padding default=valid
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=2)) #pool_size default=2
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), padding='same', activation='relu')) #padding default=valid
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=2)) #pool_size default=2
model.add(Dropout(0.2))


model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax')) #ouput 





#3. 컴파일, 훈련


# modelpath = './model/cifar-10_CNN-{epoch:02d}-{val_loss:.4f}.hdf5' 



from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# cp = ModelCheckpoint(filepath=modelpath, 
#                      monitor='val_loss', 
#                      save_best_only=True, 
#                      mode='auto'
# )

early_stopping = EarlyStopping(monitor='loss', patience=10, mode='auto')

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

hist = model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=1,
          validation_split=0.2, callbacks=[early_stopping])


#모델 저장
model.save('./save/cifar10_cnn_model.h5')

#가중치 저장
model.save_weights('./save/cifar10_cnn_weights.h5')


#4. 평가, 예측
result = model.evaluate(x_test, y_test, batch_size=32)


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




print("======cifar10_CNN=======")

model.summary()


print("loss: ", loss)
print("acc: ", accuracy)


#정답
y_answer = np.argmax(y_answer, axis=1)

#예측값
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)

print("예측값: ", y_predict)
print("정답: ", y_answer)


'''
======cifar10_CNN=======
Model: "sequential_22"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_18 (Conv2D)           (None, 32, 32, 32)        896       
_________________________________________________________________
conv2d_19 (Conv2D)           (None, 32, 32, 32)        9248      
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 16, 16, 32)        0         
_________________________________________________________________
dropout_14 (Dropout)         (None, 16, 16, 32)        0         
_________________________________________________________________
conv2d_20 (Conv2D)           (None, 16, 16, 64)        18496     
_________________________________________________________________
conv2d_21 (Conv2D)           (None, 16, 16, 64)        36928     
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 8, 8, 64)          0         
_________________________________________________________________
dropout_15 (Dropout)         (None, 8, 8, 64)          0         
_________________________________________________________________
conv2d_22 (Conv2D)           (None, 8, 8, 128)         73856     
_________________________________________________________________
conv2d_23 (Conv2D)           (None, 8, 8, 128)         147584    
_________________________________________________________________
max_pooling2d_5 (MaxPooling2 (None, 4, 4, 128)         0         
_________________________________________________________________
dropout_16 (Dropout)         (None, 4, 4, 128)         0         
_________________________________________________________________
flatten_3 (Flatten)          (None, 2048)              0         
_________________________________________________________________
dense_125 (Dense)            (None, 10)                20490     
=================================================================
Total params: 307,498
Trainable params: 307,498
Non-trainable params: 0
_________________________________________________________________
loss:  0.8989211916923523
acc:  0.7853999733924866
예측값:  [6 9 9 4 1 1 2 7 8 3]
정답:  [6 9 9 4 1 1 2 7 8 3]
'''




'''
======cifar10_CNN=======
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 32, 32, 128)       3584
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 30, 30, 128)       147584
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 15, 15, 128)       0
_________________________________________________________________
dropout (Dropout)            (None, 15, 15, 128)       0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 15, 15, 256)       295168
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 13, 13, 256)       590080
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 6, 6, 256)         0
_________________________________________________________________
dropout_1 (Dropout)          (None, 6, 6, 256)         0
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 6, 6, 512)         1180160
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 4, 4, 512)         2359808
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 2, 2, 512)         0
_________________________________________________________________
dropout_2 (Dropout)          (None, 2, 2, 512)         0
_________________________________________________________________
flatten (Flatten)            (None, 2048)              0
_________________________________________________________________
dense (Dense)                (None, 1024)              2098176
_________________________________________________________________
dropout_3 (Dropout)          (None, 1024)              0
_________________________________________________________________
dense_1 (Dense)              (None, 10)                10250
=================================================================
Total params: 6,684,810
Trainable params: 6,684,810
Non-trainable params: 0
_________________________________________________________________
loss:  0.7441360950469971
acc:  0.781000018119812
예측값:  [6 9 9 4 1 1 2 7 8 3]
정답:  [6 9 9 4 1 1 2 7 8 3]
'''



'''
loss:  0.7228475213050842
acc:  0.7699000239372253
'''