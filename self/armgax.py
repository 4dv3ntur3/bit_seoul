import numpy as np

x = np.array([[0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]]).astype('float32')

print(x.shape)
x = np.argmax(x, axis=-1)
print(x)


y = np.array([[0, 1, 2],
            [2, 1, 0]])
      
print(y.shape)

y = np.argmax(y, axis=-1)
print(y)