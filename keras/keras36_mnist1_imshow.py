#2020-11-16 (6일차)
#MNIST: 손글씨 데이터 
#One Hot Encoding: sklearn, tensorflow.keras

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist #텐서플로우에서 제공해 준다(수치로 변환해서 제공)

#train_test_split 할 필요 없이 알아서 나눠 준다
(x_train, y_train), (x_test, y_test) = mnist.load_data() #괄호 주의


print(x_train.shape, x_test.shape) #train에 6만장, test에 1만장 = 7만장, 아직 color channel은 없음 추후 reshape
print(y_train.shape, y_test.shape)


print(x_train[-15000])
print(y_train[-15000]) #label = 5 (정답)

plt.imshow(x_train[-15000], 'gray')
plt.show()


#8은 2보다 4배의 가치? 3은 1보다 3배의 가치? no
#One-Hot Encoder

