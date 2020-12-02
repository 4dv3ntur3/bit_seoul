#2020-12-02
#softmax: 여러 개 중에 하나 고르는 것 

import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

x = np.arange(1, 5)
y = softmax(x)


ratio = y
labels = y

#동그란 그래프(파이)
#가장 높은 걸 골라 줌 
plt.pie(ratio, labels=labels, shadow=True, startangle=90)
plt.show()