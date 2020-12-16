#2020-12-16
#Auto Encoder + MNIST

import numpy as np
from tensorflow.keras.datasets import mnist

# (x_train, y_train), (x_test, y_test) = mnist.load_data() #(60000, 28, 28) (60000,)
(x_train, _), (x_test, _) = mnist.load_data() # y값 안 가져옴 (이미지는 가져오되 이미지에 대한 라벨이 없는 상태)


# 두 방법 모두 상관없음 
x_train = x_train.reshape(60000, 784).astype('float32')/255
x_test = x_test.reshape(10000, 784)/255.

# print(x_train[0])
# print(x_test[0])

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

input_img = Input(shape=(784,))
encoded = Dense(64, activation='relu')(input_img) #input_img -> encoded: PCA -> Deep Learning으로 PCA 구현 가능
decoded = Dense(784, activation='sigmoid')(encoded) # y

autoencoder = Model(input_img, decoded)

# autoencoder.summary()

'''
Model: "functional_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 784)]             0
_________________________________________________________________
dense (Dense)                (None, 64)                50240
_________________________________________________________________
dense_1 (Dense)              (None, 784)               50960
=================================================================
Total params: 101,200
Trainable params: 101,200
Non-trainable params: 0
_________________________________________________________________
'''


# autoencoder.compile(optimizer='adam', loss='mse') #왜 sigmoid인데 loss가 mse? 근데 왜 잘 나오지? 
autoencoder.compile(optimizer='adam', loss='binary_crossentropy') #왜 sigmoid인데 loss가 mse? 근데 왜 잘 나오지? 
                                                                  #똑같이 잘 나옴... 나오는 결과값에 따라서 (연산하는 거에 따라서) 다름 
                                                                  #sigmoid라고 binary 할 건 없다 -> sigmoid: 0과 1 사이 (0이랑 1 아님!!!)
                                                                  #sigmoid만 layer 다섯 번 쌓으면 0 됨... 그래서 다른 activation이 나왔다 


#x로 x 확인
autoencoder.fit(x_train, x_train, epochs=50, batch_size=256,
                validation_split=0.2)

#x_test를 넣었을 때 x_test가 정상적으로 나오면 잘된 것
#차원축소 후 증폭하는 개념 
decoded_img = autoencoder.predict(x_test)

# 그림 확인
import matplotlib.pyplot as plt
n = 10
plt.figure(figsize=(20, 4))

for i in range(n):
    #위 
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28, 28)) #원래 test
    plt.gray()

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    #아래
    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded_img[i].reshape(28, 28)) #축소 후 증폭된 test
    plt.gray()

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()

# PCA의 또 다른 특성 -> 특성 추출
# 중요한 부분 빼고는 어느 정도 지워질 수 있다 (필요없는 놈들 제거)
# 즉 자세히 보면 좀 달라져 있을 수도 (maxpooling처럼)


