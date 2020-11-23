#2020-11-23 (11일차)
#Machine Learning: XOR

from sklearn.svm import LinearSVC 
from sklearn.metrics import accuracy_score #분류모델. 회귀모델에서는 r2_score


#1. 데이터
x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_data = [0, 1, 1, 0]

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
[[0, 0], [0, 1], [1, 0], [1, 1]] 의 예측 결과:  [1 1 1 1]
acc 1:  0.5
acc 2 :  0.5

[[0, 0], [0, 1], [1, 0], [1, 1]] 의 예측 결과:  [0 1 1 1]
acc 1:  0.75
acc 2 :  0.75


-> 0.5와 0.75 왔다갔다 함: 인공지능의 겨울 -> 2차 함수!
새로운 모델을 쓰면 된다

'''

