#2020-11-18 (8일차)
#load_breast_cancer -> DNN: checkpoints / model.fit() 이후 model.save() / model.save_weights()
#유방암 데이터 -> 걸렸는지 / 안 걸렸는지 -> DNN
#Classes 2

#2진 분류

#sigmoid(relu, softmax 이전에 나왔던 것) - MinMaxScaler와 유사
# 함수값이 (0, 1)로 제한된다.
# 중간 값은 1/2이다.
# 매우 큰 값을 가지면 함수값은 거의 1이며, 매우 작은 값을 가지면 거의 0이다.

#단점
# 1) Gradient Vanishing 현상:
# 미분함수에 대해 x=0에서 최대값 1/4 을 가지고, input값이 일정이상 올라가면 미분값이 거의 0에 수렴하게된다. 
# 이는 |x|값이 커질 수록 Gradient Backpropagation시 미분값이 소실될 가능성이 크다.
# 2) 함수값 중심이 0이 아니다. : 함수값 중심이 0이 아니라 학습이 느려질 수 있다. 
# 만약 모든 x값들이 같은 부호(ex. for all x is positive) 라고 가정하고 아래의 파라미터 w에 대한 미분함수식을 살펴보자. 
# ∂L∂w=∂L∂a∂a∂w 그리고 ∂a∂w=x이기 때문에, ∂L∂w=∂L∂ax 이다. 위 식에서 모든 x가 양수라면 결국 ∂L∂w는 ∂L∂a 부호에 의해 결정된다.
# 따라서 한 노드에 대해 모든 파라미터w의 미분값은 모두 같은 부호를 같게된다. 
# 따라서 같은 방향으로 update되는데 이러한 과정은 학습을 zigzag 형태로 만들어 느리게 만드는 원인이 된다.
# exp 함수 사용시 비용이 크다.



import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, LSTM
from tensorflow.keras.layers import Flatten, MaxPooling2D, Dropout



#1. 데이터 
from sklearn.datasets import load_breast_cancer

datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

print(x.shape) #(569, 30)
print(y.shape) #(569,)




#전처리
from sklearn.model_selection import train_test_split 

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7
)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(x_train)


x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# print(x_train.shape)


x_predict = x_train[30:40]
y_answer = y_train[30:40]


#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv2D, MaxPooling2D, Flatten #LSTM도 layer


model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(x_train.shape[1],))) #flatten하면서 곱하고 dense에서 또 100 곱함 
                                        #Conv2d의 activatio n default='relu'
                                        #LSTM의 activation default='tanh'
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(70, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid')) #2진분류: sigmoid -> output: 0 or 1 이니까 1개임 




#3. 컴파일 및 훈련
modelpath = './model/cancer_dnn_{epoch:02d}-{val_loss:.4f}.hdf5' 

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

cp = ModelCheckpoint(filepath=modelpath, 
                     monitor='val_loss', 
                     save_best_only=True, 
                     mode='auto'
)
early_stopping = EarlyStopping(monitor='loss', patience=10, mode='auto')

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

hist = model.fit(
    x_train,
    y_train,
    callbacks=[early_stopping, cp],
    validation_split=0.2,
    epochs=100, batch_size=32
)

#모델+가중치
model.save('./save/cancer_dnn_model_weigths.h5')

#가중치
model.save_weights('./save/cancer_dnn_weights.h5')


#4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test, batch_size=32)

print("=======cancer_dnn=======")
model.summary()
print("loss: ", loss)
print("acc: ", accuracy)


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


# #정답

# #예측값
# y_predict = model.predict(x_predict)
# y_predict = y_predict.reshape(10,)

# print("예측값: ", y_predict)
# print("정답: ", y_answer)

'''
=======cancer_dnn=======
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
dense_5 (Dense)              (None, 1024)              308224
_________________________________________________________________
dense_6 (Dense)              (None, 150)               153750
_________________________________________________________________
dense_7 (Dense)              (None, 70)                10570
_________________________________________________________________
dense_8 (Dense)              (None, 32)                2272
_________________________________________________________________
dense_9 (Dense)              (None, 1)                 33
=================================================================
Total params: 713,037
Trainable params: 713,037
Non-trainable params: 0
_________________________________________________________________
loss:  0.08616919070482254
acc:  0.9766082167625427
'''