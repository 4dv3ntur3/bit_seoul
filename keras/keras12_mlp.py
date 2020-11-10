
# Multi Layer Perceptron

#1. 데이터
import numpy as np

x = np.array((range(1, 101), range(711, 811), range(100)))
y = np.array((range(101, 201), range(311, 411), range(100)))


# print(x)
# print(x.shape) # (3, )

# (100, 3)으로 변환해야 한다
x = np.transpose(x)
y = np.transpose(y)

print(x.shape)
print(y.shape)

for i in range(0, 50):
    print(i+1, ". ", "x: ", x[i], "\n", "   y: ", y[i])
    
