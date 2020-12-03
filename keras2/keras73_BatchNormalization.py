#2020-12-02 (18일차)
#kernel_regularizer
#bias_regularizer
#batch_normalization



import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist #텐서플로우에서 제공해 준다(수치로 변환해서 제공)

(x_train, y_train), (x_test, y_test) = mnist.load_data() #괄호 주의


#1. 데이터 전처리: OneHotEncoding 대상은 Y
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train)


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

from tensorflow.keras.layers import BatchNormalization, Activation, Dropout
from tensorflow.keras.regularizers import l1, l2, l1_l2 #1과 2를 섞은 것 

model = Sequential()
model.add(Conv2D(30, (2, 2), padding='same', input_shape=(28, 28, 1))) #padding 주의!

#######뭐가 맞다, 아니다가 아님. 할 수 있다
#######맞다, 아니다는 결과치를 보고 결정 
model.add(BatchNormalization()) #normalization
model.add(Activation('relu'))

model.add(Conv2D(50, (2, 2), kernel_initializer='he_normal'))
model.add(BatchNormalization()) #normalization
model.add(Activation('relu'))

model.add(Conv2D(30, (2, 2), kernel_regularizer=l1(l1=0.01)))
model.add(Dropout(0.2))



model.add(MaxPooling2D(pool_size=2)) #pool_size default=2
model.add(Flatten()) 
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='softmax'))



#3. 컴파일, 훈련
#원래 loss=mse지만 다중분류에서는 반드시 loss='categorical_crossentropy'                     
#OneHotEncoding -> output layer activation=softmax -> loss='categorical_crossentropy: #10개를 모두 합치면 1이 되는데, 가장 큰 값의 위치가 정답    

#tensorboard 
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

early_stopping = EarlyStopping(monitor='loss', patience=30, mode='auto')

#log가 들어갈 폴더='graph'
#여기까지 해서 graph 폴더 생기고 자료들 들어가 있으면 텐서보드 쓸 준비 ok
#단, 로그가 많으면 겹쳐서 보일 수 있으니 그럴 땐 로그 삭제하고 

model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy']) #"mean_squared_error" (풀네임도 가능하다)

model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=1,
          validation_split=0.2, callbacks=[early_stopping])



#4. 평가, 예측
#fit에서 쓴 이름과 맞춰 주기 

loss, accuracy = model.evaluate(x_test, y_test, batch_size=32)


print("=======mnist_CNN=======")
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



'''
=======mnist_CNN=======
loss:  0.11501652747392654
acc:  0.9843999743461609
예측값:  [4 0 9 1 1 2 4 3 2 7]
정답:  [4 0 9 1 1 2 4 3 2 7]
'''








#경사 소실과 경사 폭주
#gradient vanishing & gradient exploding































#남들의 튜닝 잘된 것을 보고 베끼는 것도 좋은 공부가 됨 (빨리 는다)





# Regularization은 W(weight)가 너무 큰 값들을 갖지 않도록 하는 것을 말합니다. 값이 커지게 되면 구불구불한 형태의 코스트함수가 만들어지고 예측에 실패하게 되는데, 이를 머신러닝에서는 “데이터보다 모델의 복잡도(Complexity)가 크다” 라고 설명합니다.
# 과도하게 복잡하기 때문에 발생하는 문제라고 보는 것이고, 이를 낮추기 위한 방법이 Regularization입니다.

# 가중치가 클수록 큰 패널티를 부과하여 Overfitting을 억제하는 방법

# Regularization에는 L1정규화, L2정규화가 있는데 둘의 차이는 다음과 같습니다.

# L1 regularization : 대부분의 요소값이 0인 sparse feature에 의존한 모델에서 불필요한 feature에 대응하는 가중치를 0으로 만들어 해당 feature를 모델이 무시하게 만듬. feature selection효과가 있다.
# L2 regularization : 아주 큰 값이나 작은 값을 가지는 outlier모델 가중치에 대해 0에 가깝지만 0은 아닌값으로 만듬. 선형모델의 일반화능력을 개선시키는 효과가 있다.
# –> 패널티에 대한 효과를 크게보기 위해 L1보다 L2를 많이 사용하는 경향이 있음.

'''
activation: 모든 연산의 마지막!
kernel= weight / bias

결과값이 아니라 가중치 값!!!(w값)
kernel_regularization: 가중치 제한 L1 (절대값 규제. 음수) / L2 (제곱 규제. 이것도 음수 문제 해결)

activation은 y값 
통과해서 batchnormalizataion 

사실 성능에서 큰 효과를 보는 건 데이터 전처리이다...



kernel_initializer: 가중치 초기화(!= 규제)
각 뉴런은 특정한 가중치로 초기화할 수 있다.
케라스는 몇 가지 선택 사항을 제공하며, 일반적으로 사용하는 것은 다음과 같다.

1. kernel_initializer = "random_uniform" : 가중치는 -0.05 ~ 0.05로 균등하게 작은 임의의 값으로 초기화한다.

2. kernel_initializer = "random_normal" : 가중치는 평균이 0이고, 표준편차가 0.05로 정규분포에 따라 초기화한다.

3. kernel_initializer = "zero" : 모든 가중치를 0으로 초기화한다.
'''