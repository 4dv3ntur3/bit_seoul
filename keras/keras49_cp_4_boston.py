#2020-11-18 (8일차)
#Boston -> DNN + ModelCheckPoint
#사이킷런의 dataset

'''
x
506 행 13 열 
CRIM     per capita crime rate by town
ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
INDUS    proportion of non-retail business acres per town
CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
NOX      nitric oxides concentration (parts per 10 million)
RM       average number of rooms per dwelling
AGE      proportion of owner-occupied units built prior to 1940
DIS      weighted distances to five Boston employment centres
RAD      index of accessibility to radial highways
TAX      full-value property-tax rate per $10,000
PTRATIO  pupil-teacher ratio by town
B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
LSTAT    % lower status of the population

y
506 행 1 열
target (MEDV)     Median value of owner-occupied homes in $1000's (집값)
'''


from numpy import array
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input


#1. 데이터
from sklearn.datasets import load_boston
dataset = load_boston() #data(X)와 target(Y)으로 구분되어 있다
x = dataset.data
y = dataset.target

print("x: ", x)
print("y: ", y) #전처리가 되어 있지 않다

print(x.shape) #(506, 13)
print(y.shape) #(506,)


#전처리: 수치가 크다

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7
)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(x_train)
#train

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# x_pred = scaler.transform(x_pred)


#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout #LSTM도 layer


model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(13,))) #default activation = linear

#hidden layer
model.add(Dense(256, activation='relu'))
model.add(Dense(400, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(1500, activation='relu'))
model.add(Dense(750, activation='relu'))
model.add(Dense(400, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1)) #output




#3. 컴파일 및 훈련
modelpath = './model/boston_dnn-{epoch:02d}-{val_loss:.4f}.hdf5' 

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

cp = ModelCheckpoint(filepath=modelpath, 
                     monitor='val_loss', 
                     save_best_only=True, 
                     mode='auto')


early_stopping = EarlyStopping(monitor='loss', patience=10, mode='auto')

model.compile(loss='mse', optimizer='adam', metrics=['mse'])

hist = model.fit(
    x_train,
    y_train,
    callbacks=[early_stopping, cp],
    validation_split=0.2,
    epochs=100, batch_size=10
)

# #fit에 있는 네 가지
# loss = hist.history['loss']
# val_loss = hist.history['val_loss']
# acc = hist.history['mse']
# val_acc = hist.history['val_mse']


#모델+가중치
model.save('./save/boston_dnn_model_weights.h5')

model.save_weights('./save/boston_dnn_weights.h5')


#4. 평가, 예측


result = model.evaluate(x_test, y_test, batch_size=10)

print("=====boston_DNN=====")
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
=====boston_DNN=====
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 64)                896
_________________________________________________________________
dense_1 (Dense)              (None, 256)               16640
_________________________________________________________________
dense_2 (Dense)              (None, 400)               102800
_________________________________________________________________
dense_3 (Dense)              (None, 512)               205312
_________________________________________________________________
dense_4 (Dense)              (None, 1500)              769500
_________________________________________________________________
dense_5 (Dense)              (None, 750)               1125750
_________________________________________________________________
dense_6 (Dense)              (None, 400)               300400
_________________________________________________________________
dense_7 (Dense)              (None, 150)               60150
_________________________________________________________________
dense_8 (Dense)              (None, 64)                9664
_________________________________________________________________
dense_9 (Dense)              (None, 1)                 65
=================================================================
Total params: 2,591,177
Trainable params: 2,591,177
Non-trainable params: 0
_________________________________________________________________
loss, mse:  16.289348602294922 16.289348602294922
RMSE:  4.036006545064242
R2:  0.7942954440283122
'''