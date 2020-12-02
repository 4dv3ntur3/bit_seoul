#2020-12-02
#LSTM의 default acitvation= tanh

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)
y = np.tanh(x)

plt.plot(x, y)
plt.grid()
plt.show() 


#tanh을 통과하게 되면 -1에서 1 사이로 수렴한다 
