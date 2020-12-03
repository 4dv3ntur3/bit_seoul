#2020-12-02 (18일차)
#sigmoid
#소스 리폼은 얼마든지 환영 

import matplotlib.pyplot as plt
import numpy as np

def sigmoid(x):
    #값이 0과 1 사이로 수렴 (0 아니면 1이 아니라)
    return 1 / (1 + np.exp(-x))

x = np.arange(-5, 5, 0.1)
y = sigmoid(x)

print("x: ", x)
print("y: ", y)

plt.plot(x, y)
plt.grid()
plt.show()

