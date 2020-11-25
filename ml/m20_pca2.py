#2020-11-25 (13일차)
#PCA

import numpy as np
from sklearn.decomposition import PCA #decomposition:분해
from sklearn.datasets import load_diabetes

datasets = load_diabetes()

x = datasets.data
y = datasets.target

print(x.shape) #(442, 10)
print(y.shape) #(442,)


pca = PCA()
pca.fit(x) #fit만 됨

#누적된 합을 표시하겠다
cumsum = np.cumsum(pca.explained_variance_ratio_) #축소된 차원들의 가치의 합
print(cumsum) #[0.40242142 0.55165324 0.67224947 0.76779711 0.83401567 0.89428759
               # 0.94794364 0.99131196 0.99914395 1.        ] : 1개로 축소했을 때, 2개로 축소했을 때... 10개로 축소했을 때(그대로)는 1

d = np.argmax(cumsum >= 0.95)  + 1 #끝에 세 개가 골라지겠지 index를 반환하므로 1을 더한다
print(cumsum >= 0.95) #0.95보다 크니? 
print(d) #8
#코드 익숙 + 알고리즘 이해



import matplotlib.pyplot as plt

plt.plot(cumsum)
plt.grid() #격자에 넣어 줌
plt.show()



#당뇨병으로 했을 때 0인 놈들이 많았음 -> 빼 버리면 되잖아







#pca라는 모델 만든다
#y 없다고 생각하고 x만 있다고 생각
'''
pca = PCA(n_components=10) #n_components 축소할 컬럼의 수. 기존 차원보다 많으면 안 됨.
                          #MNIST의 경우 shape가 784, 인데 이걸 축소해서 넣을 수도 있겠지 다만 데이터의 손실도 있을 수 있다
                          #하지만 그걸 감수하고서라도 속도를 잡을 수 있음
x2d = pca.fit_transform(x) #fit_transform은 한 방에 fit과 transform이 같이 된다

print(x2d.shape) #(442, n_components)

pca_EVR = pca.explained_variance_ratio_ #explaiend variance ratio: feature importance (축소된 차원들의 가치) 
print(pca_EVR) #n_components만큼 나온다 [0.40242142 0.14923182] 
print(sum(pca_EVR)) #n_components 5보다 7일 때가 더 높은 값 나옴. 차원을 너무 축소해서도 안 됨. (합) 하지만 0.947이니까 5퍼센트 정도 손실률 있음
                    #얘도 100프로 신뢰는 x (기준값으로 사용할 뿐)

#이 데이터셋을 신뢰할 수 없ㅇ므로 90퍼센트만 가지고 작업하겠다 -> 7 했을 때 0.94 나왔으니까 7로 해놓고 씀
'''
