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



# from tensorflow.keras.models import load_model
# model = load_model('./save/model_test02_2.h5')





#3. 컴파일, 훈련
#4. 평가, 예측

#얘네는 evaluate에서 나온 애들

result = model.evaluate(x_test, y_test, batch_size=32)
print("loss: ", result[0])
print("acc: ", result[1])


#keras51_1 의 결과값과 같다 (거기서 model save를 했으니까)
# loss:  0.08255356550216675
# acc:  0.980400025844574