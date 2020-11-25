#2020-11-25 (13일차)
#PCA: mnist

import numpy as np
from tensorflow.keras.datasets import mnist

from sklearn.decomposition import PCA #decomposition:분해

#x만 빼겠다
(x_train, _), (x_test, _) = mnist.load_data()


# print(x_train.shape) 
# print(x_test.shape) 


x = np.append(x_train, x_test, axis=0)
print(x.shape) 

x = x.reshape(x.shape[0], x.shape[1]*x.shape[2]) #(70000, 28, 28)



pca = PCA()
pca.fit(x) #fit만 됨

#누적된 합을 표시하겠다
cumsum = np.cumsum(pca.explained_variance_ratio_) #축소된 차원들의 가치의 합
d = np.argmax(cumsum >= 1)  + 1 #index 반환이므로 개수 확인은 +1 (0.95 이상이 되는 최소의 column 수 반환인 것 같다)
#즉, d는 우리가 필요한 n_components의 수
# print(cumsum >= 0.9) #T/F 확인
print(d) 

#실습
#pca를 통해 마구마구 0.95 이상인 게 몇 개?
#pca 배운 거 다 집어넣고 확인

#0.95 -> 154
#0.9 -> 87
#0.8 -> 43
#0.5 -> 11
#0.1 -> 2
#1 -> 713 : 784-713 = 71개는 필요가 없는 column임
#코드 익숙 + 알고리즘 이해










