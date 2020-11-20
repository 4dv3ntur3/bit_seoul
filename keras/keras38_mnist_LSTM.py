#2020-11-17 (7일차)
#MNIST -> LSTM
#파일 하단에 결과치 기록할 것(loss, acc)
#CNN vs DNN vs LSTM

#epoch가 10회 돌았는데도 acc가 0.1이다 -> 그럼 모델 다시 살펴보기
#초반에 가파르게 상승해야 함 (너무 미적미적대면 아님)


#(60000, 28, 28) -> (60000, ?) 행은 건드리면 안 됨 (데이터 개수니까. 어차피 행 무시)


import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist #텐서플로우에서 제공해 준다(수치로 변환해서 제공)

#train_test_split 할 필요 없이 알아서 나눠 준다
(x_train, y_train), (x_test, y_test) = mnist.load_data() #괄호 주의

#60000장 * 28pixel * 28pixel
# print(x_train.shape, x_test.shape) #(60000, 28, 28)(10000, 28, 28)
# print(y_train.shape, y_test.shape) #(60000, )      (10000,)        : 스칼라


# print(x_train[0])
# print(y_train[1]) #label 


# plt.imshow(x_train[0], 'gray')
# plt.show()


#8은 2보다 4배의 가치? 3은 1보다 3배의 가치? no
#One-Hot Encoder
#y_train: 60000, -> OneHotEncoding : 1 0 0 0 0 0 0 0 0 0 (60000, 10) : 분류가 10개니까 (0~9)



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

x_train = x_train.reshape(60000, 28, 28).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28).astype('float32')/255.
                        #x_test.shape[0], x_test.shape[1] ... 



#predict data, answer data
x_predict = x_train[20:30]
y_answer = y_train[20:30]



#CNN에 넣을 수 있는 4차원 reshape + y도 onehotencoding
#scaler 사용해야: 어떤 게 더 좋을지는 해 봐야 안다
#지금 이 상황에서 M은 255라는 걸 알고 있음. 그러므로 MinMax에서는 255로 나누면 0~1 사이로 수렴 가능


# print(x_train[0]) 


#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, LSTM

model = Sequential()
model.add(LSTM(1000, activation='relu', input_shape=(28, 28))) #꼭 28, 28, 1일 필요는 없음 #뭔가 시계열 같은 데이터라고 판단이 되면 몇 개씩 자를지 생각할 수도 있음
model.add(Dense(500, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(50, activation='relu'))
# model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='softmax')) #softmax** : 2 이상 분류(다중분류)의 activation은 softmax, 2진분류는 sigmoid(여자/남자, dead/alive)
                                            #즉 softmax를 사용하려면 OneHotEncoding 해야



#3. 컴파일, 훈련
#원래 loss=mse지만 다중분류에서는 반드시 loss='categorical_crossentropy'                     
#OneHotEncoding -> output layer activation=softmax -> loss='categorical_crossentropy: #10개를 모두 합치면 1이 되는데, 가장 큰 값의 위치가 정답    

#tensorboard 
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

early_stopping = EarlyStopping(monitor='loss', patience=8, mode='auto') 

#log가 들어갈 폴더='graph'
#여기까지 해서 graph 폴더 생기고 자료들 들어가 있으면 텐서보드 쓸 준비 ok
#단, 로그가 많으면 겹쳐서 보일 수 있으니 그럴 땐 로그 삭제하고 
to_hist = TensorBoard(log_dir='graph', histogram_freq=0,
                      write_graph=True, write_images=True
)

model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy']) #"mean_squared_error" (풀네임도 가능하다)

model.fit(x_train, y_train, epochs=100, batch_size=100, validation_split=0.2,
                            callbacks=[early_stopping])



#4. 평가, 예측
#fit에서 쓴 이름과 맞춰 주기 

loss, accuracy = model.evaluate(x_test, y_test, batch_size=32)



print("=======mnist_LSTM=======")

print("loss: ", loss)
print("acc: ", accuracy)


#y값 원상복구 
#np.argmax(, axis=1)


#정답
y_answer = np.argmax(y_answer, axis=1)

#예측값
y_predict = model.predict(x_predict)
y_predict = np.argmax(y_predict, axis=1)



model.summary()
print("예측값: ", y_predict)
print("정답: ", y_answer)


'''
loss:  0.09111423045396805
acc:  0.9911999702453613
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lstm (LSTM)                  (None, 1000)              4116000
_________________________________________________________________
dense (Dense)                (None, 500)               500500
_________________________________________________________________
dense_1 (Dense)              (None, 300)               150300
_________________________________________________________________
dense_2 (Dense)              (None, 200)               60200
_________________________________________________________________
dense_3 (Dense)              (None, 50)                10050
_________________________________________________________________
dense_4 (Dense)              (None, 10)                510 

=================================================================
Total params: 4,837,560
Trainable params: 4,837,560
Non-trainable params: 0
_________________________________________________________________
예측값:  [4 0 9 1 1 2 4 3 2 7]
정답:  [4 0 9 1 1 2 4 3 2 7]
'''