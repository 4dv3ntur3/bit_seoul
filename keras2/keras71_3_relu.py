#2020-12-02 (18일차)
#relu

import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

x = np.arange(-5, 5, 0.1)
y = relu(x)

plt.plot(x, y)
plt.grid()
plt.show()

#relu 친구들 찾기
#relu -> Leaky ReLu, PReLU: dead ReLU 해결 -> ELU: 아웃라이어 음수쪽 saturation 시킴 -> SeLU(scaled exponential linear unit)
