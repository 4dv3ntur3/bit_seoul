#2020-12-02 (18일차)
#gradient 

import numpy as np
import matplotlib.pyplot as plt

f = lambda x: x**2 -4*x + 6 #2차 함수
x = np.linspace(-1, 6, 100)

y = f(x)


# 그리잣
plt.plot(x, y, 'k-')
plt.plot(2, 2, 'sk')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.show()