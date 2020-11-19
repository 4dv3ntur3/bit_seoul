#2020-11-19
#LSTM vs Conv1d: cifar-100


from tensorflow.keras.datasets import cifar100

#이미지 분류-> OneHotEncoding
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, LSTM
from tensorflow.keras.layers import Flatten, MaxPooling2D #maxpooling2d는 들어가도 되고 안 들어가도 됨 필수 아님
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar100.load_data()



#1. 데이터 전처리: OneHotEncoding 대상은 Y
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# print(y_train)


# print(y_train.shape, y_test.shape)
# print(y_train[0]) #y_train[0]=5 -> [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]


#shape 바꿀 줄 알아야 함
#60000, 14, 14, 4도 가능하고 60000, 28, 14, 2도 가능
#LSTM으로도 바꿀 수 있다

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2]*3).astype('float32')/255.
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2]*3).astype('float32')/255.
                        #x_test.shape[0], x_test.shape[1] ... 



#CNN에 넣을 수 있는 4차원 reshape + y도 onehotencoding
#scaler 사용해야: 어떤 게 더 좋을지는 해 봐야 안다
#지금 이 상황에서 M은 255라는 걸 알고 있음. 그러므로 MinMax에서는 255로 나누면 0~1 사이로 수렴 가능





#predict data, answer data
x_predict = x_train[20:30]
y_answer = y_train[20:30]



# print(x_train.shape)


#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout

model = Sequential()
model.add(Conv1D(512, 3, activation='relu', padding='same', input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(Conv1D(512, 3, activation='relu', padding='same'))
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(256, 3, padding='same', activation='relu'))
model.add(Conv1D(256, 3, activation='relu', padding='same'))
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(128, 3, padding='same', activation='relu'))
model.add(Conv1D(128, 3, activation='relu', padding='same'))
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(64, 3, padding='same', activation='relu'))
model.add(Conv1D(64, 3, activation='relu', padding='same'))
model.add(MaxPooling1D(pool_size=2))


model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='softmax')) #ouput 



#3. 컴파일, 훈련
#원래 loss=mse지만 다중분류에서는 반드시 loss='categorical_crossentropy'                     
#OneHotEncoding -> output layer activation=softmax -> loss='categorical_crossentropy: #10개를 모두 합치면 1이 되는데, 가장 큰 값의 위치가 정답    

#tensorboard 
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

early_stopping = EarlyStopping(monitor='loss', patience=10, mode='auto')

#log가 들어갈 폴더='graph'
#여기까지 해서 graph 폴더 생기고 자료들 들어가 있으면 텐서보드 쓸 준비 ok
#단, 로그가 많으면 겹쳐서 보일 수 있으니 그럴 땐 로그 삭제하고 
to_hist = TensorBoard(log_dir='graph', histogram_freq=0,
                      write_graph=True, write_images=True
)

model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy']) #"mean_squared_error" (풀네임도 가능하다)

model.fit(x_train, y_train, epochs=100, batch_size=512, validation_split=0.2, callbacks=[early_stopping])



#4. 평가, 예측
#fit에서 쓴 이름과 맞춰 주기 

loss, accuracy = model.evaluate(x_test, y_test, batch_size=512)

print("======cifar100_conv1d=======")

print("loss: ", loss)
print("acc: ", accuracy)


#y값 원상복구 
#np.argmax(, axis=1)


#정답
y_answer = np.argmax(y_answer, axis=1)

#예측값
y_predict = model.predict(x_predict)
y_predict = np.argmax(y_predict, axis=1)


print("예측값: ", y_predict)
print("정답: ", y_answer)
model.summary()


'''

======cifar100_conv1d=======
loss:  5.837181091308594
acc:  0.2840000092983246
예측값:  [79 59 70 87 59 84 64 52 42 64]
정답:  [74 59 70 87 59 84 64 52 42 64]
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv1d (Conv1D)              (None, 32, 512)           147968    
_________________________________________________________________
conv1d_1 (Conv1D)            (None, 32, 512)           786944
_________________________________________________________________
max_pooling1d (MaxPooling1D) (None, 16, 512)           0
_________________________________________________________________
conv1d_2 (Conv1D)            (None, 16, 256)           393472
_________________________________________________________________
conv1d_3 (Conv1D)            (None, 16, 256)           196864
_________________________________________________________________
max_pooling1d_1 (MaxPooling1 (None, 8, 256)            0
_________________________________________________________________
conv1d_4 (Conv1D)            (None, 8, 128)            98432
_________________________________________________________________
conv1d_5 (Conv1D)            (None, 8, 128)            49280
_________________________________________________________________
max_pooling1d_2 (MaxPooling1 (None, 4, 128)            0
_________________________________________________________________
conv1d_6 (Conv1D)            (None, 4, 64)             24640
_________________________________________________________________
conv1d_7 (Conv1D)            (None, 4, 64)             12352
_________________________________________________________________
max_pooling1d_3 (MaxPooling1 (None, 2, 64)             0
_________________________________________________________________
flatten (Flatten)            (None, 128)               0
_________________________________________________________________
dense (Dense)                (None, 1024)              132096
_________________________________________________________________
dropout (Dropout)            (None, 1024)              0
_________________________________________________________________
dense_1 (Dense)              (None, 100)               102500
=================================================================
Total params: 1,944,548
Trainable params: 1,944,548
Non-trainable params: 0
_________________________________________________________________

'''