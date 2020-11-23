#2020-11-23 (11일차)
#Machine Learning: OR

from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score #분류모델. 회귀모델에서는 r2_score


#1. 데이터
x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_data = [0, 1, 1, 1]

#2. 모델
model = LinearSVC()

#3. 훈련 (컴파일 필요 없음)
model.fit(x_data, y_data)

#4. 평가, 예측

y_predict = model.predict(x_data) #y_data가 나오는지 보면 된다 
print(x_data, '의 예측 결과: ', y_predict)


acc_1 = model.score(x_data, y_data)
print('acc 1: ', acc_1)

acc_2 = accuracy_score(y_data, y_predict)
print("acc 2 : ", acc_2)


'''
[[0, 0], [0, 1], [1, 0], [1, 1]] 의 예측 결과:  [0 1 1 1]
acc 1:  1.0
acc 2 :  1.0
'''

