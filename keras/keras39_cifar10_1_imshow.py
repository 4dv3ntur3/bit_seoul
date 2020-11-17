#2020-11-17 (7일차)
#cifar-10
#color이므로 channel=3

from tensorflow.keras.datasets import cifar10

#이미지 분류-> OneHotEncoding
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, LSTM
from tensorflow.keras.layers import Flatten, MaxPooling2D #maxpooling2d는 들어가도 되고 안 들어가도 됨 필수 아님
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train[0])
print("y_train[0]: ", y_train[0])

#데이터 구조 확인
print(x_train.shape, x_test.shape) #(50000, 32, 32, 3) (10000, 32, 32, 3)
print(y_train.shape, y_test.shape) #(50000, 1) (10000, 1)


#우리가 볼 수 있는 이미지로 바로 확인 가능
#pixel이 32*32라서 작게 줄여야 그나마 알아볼 수 있음...
#훈련용 5만장, 테스트 1만장
plt.imshow(x_train[0])
plt.show()