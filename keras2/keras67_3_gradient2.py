#2020-12-02 (18일차)
#learning rate

import numpy as np
import matplotlib.pyplot as plt

f = lambda x: x**2 -4*x + 6 #2차 함수

#가장 밑의 최소값 찾으러 가는 여정
#미분해서 그 값이 0일 때 최소값

#f'(x) = gradient
gradient = lambda x:2*x -4 #얘가 0일 때가 최소. 

x0 = 0.0
# MaxIter = 10
MaxIter = 30

# learning_rate = 0.25
learning_rate = 0.1


print("step\tx\tf(x)")
print("{:02d}\t{:6.5f}\t{:6.5f}".format(0, x0, f(x0)))

for i in range(MaxIter): 
    x1 = x0 - gradient(x0)*learning_rate #즉 x=2일 때 근데 0->2로 가야 하는데 learning rate 빼면 숫자가 오히려 작아짐...
    x0 = x1

    print("{:02d}\t{:6.5f}\t{:6.5f}".format(i+1, x0, f(x0)))    

    #10번만에 2.0 찾았다 


#learning rate가 줄어들면 epoch(iter)를 늘려야
#계속 간격 좁혀가면서 찾아냄 