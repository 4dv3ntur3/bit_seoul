#2020-11-18 (8일차)
#모델+가중치 load


import numpy as np
from tensorflow.keras.datasets import mnist 

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data() #괄호 주의



#전처리: OneHotEncoding 대상은 Y
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.


# #predict data, answer data
# x_predict = x_train[20:30]
# y_answer = y_train[20:30]



#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten



#모델+가중치 저장했던 것 load
from tensorflow.keras.models import load_model
model = load_model('./save/model_test02_2.h5')





#3. 컴파일, 훈련

#4. 평가, 예측
result = model.evaluate(x_test, y_test, batch_size=32)
print("loss: ", result[0])
print("acc: ", result[1])


#keras51_1 의 결과값과 같다 (거기서 model save를 했으니까)
# loss:  0.08255356550216675
# acc:  0.980400025844574