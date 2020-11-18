#2020-11-18 (8일차)
#diabetes -> DNN + ModelCheckPoint
#사이킷런의 데이터셋 load_diabetes


from numpy import array
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input


#1. 데이터
from sklearn.datasets import load_diabetes
dataset = load_diabetes() #data(X)와 target(Y)으로 구분되어 있다
x = dataset.data 
y = dataset.target

# print("x: ", x)
# print("y: ", y) #전처리가 되어 있지 않다

# print(x.shape) #(442, 10)
# print(y.shape) #(442, )


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=33
)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# x_pred = scaler.transform(x_pred)


#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout #LSTM도 layer


model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(x_train.shape[1],))) #flatten하면서 곱하고 dense에서 또 100 곱함 
                                        #Conv2d의 activatio n default='relu'
                                        #LSTM의 activation default='tanh'
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='relu'))

#output _ 선형회귀 
model.add(Dense(1))


#3. 컴파일 및 훈련

modelpath = './model/diabetes_DNN-{epoch:02d}-{val_loss:.4f}.hdf5' 

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

early_stopping = EarlyStopping(monitor='loss', patience=10, mode='auto')

cp = ModelCheckpoint(filepath=modelpath, 
                     monitor='val_loss', 
                     save_best_only=True, 
                     mode='auto'
)

model.compile(loss='mse', optimizer='adam', metrics=['mse'])

hist = model.fit(
    x_train,
    y_train,
    callbacks=[early_stopping, cp],
    validation_split=0.2,
    epochs=100, batch_size=32
)

#모델, 가중치
model.save('./save/diabetes_dnn_model_weights.h5')

#가중치
model.save('./save/diabetes_dnn_weights.h5')


#4. 평가, 예측
result = model.evaluate(x_test, y_test, batch_size=32)
print("=====Diabetes_DNN=====")
model.summary()
print("loss, mse: ", result[0], result[1])

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



y_pred = model.predict(x_test)


#RMSE
from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_pred))
    # predict해서 나온 값과 원래 y_test 값을 비교해서 RMSE로 나오게 하겠다
print("RMSE: ", RMSE(y_test, y_pred))


# R2는 함수 제공
from sklearn.metrics import r2_score
print("R2: ", r2_score(y_test, y_pred))



'''
=====Diabetes_DNN=====
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 512)               5632
_________________________________________________________________
dense_1 (Dense)              (None, 256)               131328
_________________________________________________________________
dense_2 (Dense)              (None, 128)               32896
_________________________________________________________________
dropout (Dropout)            (None, 128)               0
_________________________________________________________________
dense_3 (Dense)              (None, 100)               12900
_________________________________________________________________
dense_4 (Dense)              (None, 64)                6464
_________________________________________________________________
dense_5 (Dense)              (None, 32)                2080
_________________________________________________________________
dense_6 (Dense)              (None, 10)                330
_________________________________________________________________
dense_7 (Dense)              (None, 1)                 11
=================================================================
Total params: 191,641
Trainable params: 191,641
Non-trainable params: 0
_________________________________________________________________
loss, mse:  2795.55810546875 2795.55810546875
RMSE:  52.87303868763434
R2:  0.4871686313608924

'''