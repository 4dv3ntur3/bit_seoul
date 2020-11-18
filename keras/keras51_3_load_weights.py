#2020-11-18 (8일차)
#가중치 load


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
print(y_train)


# print(y_train.shape, y_test.shape)
# print(y_train[0]) #y_train[0]=5 -> [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]


#shape 바꿀 줄 알아야 함
#60000, 14, 14, 4도 가능하고 60000, 28, 14, 2도 가능
#LSTM으로도 바꿀 수 있다

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.
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
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

model = Sequential()
model.add(Conv2D(30, (2, 2), padding='same', input_shape=(28, 28, 1))) #padding 주의!
model.add(Conv2D(50, (2, 2), padding='valid'))
model.add(Conv2D(120, (3, 3))) #padding default=valid
model.add(Conv2D(200, (2, 2), strides=2))
model.add(Conv2D(30, (2, 2)))
model.add(MaxPooling2D(pool_size=2)) #pool_size default=2
model.add(Flatten()) 
model.add(Dense(10, activation='relu')) #flatten하면서 곱하고 dense에서 또 100 곱함 
                                        #Conv2d의 activation default='relu'
                                        #LSTM의 activation default='tanh'
#MaxPooling2D-Flatten:reshape 개념

model.add(Dense(10, activation='softmax')) #softmax** : 2 이상 분류(다중분류)의 activation은 softmax, 2진분류는 sigmoid
                                            #즉 softmax를 사용하려면 OneHotEncoding 해야
# model.summary()




######모델 저장
# model.save('./save/model_test01_1.h5')








# #3. 컴파일, 훈련

# modelpath = './model/mnist-{epoch:02d}-{val_loss:.4f}.hdf5' 

# #root 폴더(=Study) 밑의 model 폴더에 
# #.hd5f라는 확장자 => *.h5 (모델 확장자와 거의 동일하다)
# #epoch:두자릿수 정수 표기-val_loss:소수 넷째자리까지 표기 -> 이 파일명으로 생성 : 파일명만 봐도 대략 어느 지점이 좋은지 알 수 있음 

# from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint

# early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='auto')

# #val_loss가 개선이 생길 때마다 history를 저장하고 싶다!
# #정의
# cp = ModelCheckpoint(filepath=modelpath, 
#                      monitor='val_loss', 
#                      save_best_only=True, 
#                      mode='auto'
# ) #계속 저장될 것 (계속 좋아질 테니까) 그러다가 튀는 구간 있으면 또 저장 안 되고... 다시 뚫고 내려가는 거 저장되고...
#   #좋은 시점 저장됨
# #model 폴더 잘 보기!
# #04-어쩌고 후에 07-어쩌고가 생겼다면 5, 6은 별로였다는 뜻


# #log가 들어갈 폴더='graph'
# #여기까지 해서 graph 폴더 생기고 자료들 들어가 있으면 텐서보드 쓸 준비 ok
# #단, 로그가 많으면 겹쳐서 보일 수 있으니 그럴 땐 로그 삭제하고 
# to_hist = TensorBoard(log_dir='graph', histogram_freq=0,
#                       write_graph=True, write_images=True
# )




model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy']) #"mean_squared_error" (풀네임도 가능하다)

# #model.fit()은 history 반환
# hist = model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=1,
#           validation_split=0.2, callbacks=[early_stopping, cp]) #tensorboard 쓰고 싶으면 여기다가 to_hist도...



# #모델 + 가중치 저장
# model.save('./save/model_test02_2.h5')

# #가중치만 저장
# model.save_weights('./save/weight_test02.h5')



# #fit에 있는 네 가지
# loss = hist.history['loss']
# val_loss = hist.history['val_loss']
# acc = hist.history['accuracy']
# val_acc = hist.history['val_accuracy']

#모델은 남겨 둬야 함
#모델 빼고 가중치만 저장하므로
#model.load_weights니까 ㅋㅋㅋㅋㅋ
model.load_weights('./save/weight_test02.h5')


#4. 평가, 예측

#얘네는 evaluate에서 나온 애들

result = model.evaluate(x_test, y_test, batch_size=32)
print("loss: ", result[0])
print("acc: ", result[1])



#정답
y_answer = np.argmax(y_answer, axis=1)

#예측값
y_predict = model.predict(x_predict)
y_predict = np.argmax(y_predict, axis=1)


print("예측값: ", y_predict)
print("정답: ", y_answer)




#결과값
'''
loss:  0.08255356550216675
acc:  0.980400025844574
예측값:  [4 0 9 1 1 2 4 3 2 7]
정답:  [4 0 9 1 1 2 4 3 2 7]

'''

#똑같다
#가중치만 저장하고 싶으면 weights_save
#잘 만든 모델이라고 판단되면 -> 
'''
loss:  0.08255356550216675
acc:  0.980400025844574
'''