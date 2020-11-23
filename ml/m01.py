#2020-11-23 (11일차)
#Machine Learning

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 10, 0.1)
y = np.sin(x) #사인 함수


plt.plot(x, y) # x 없을 땐 리스트 순서대로 x값이 됐다 
plt.show()