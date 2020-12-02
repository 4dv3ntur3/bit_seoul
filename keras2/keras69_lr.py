#2020-12-02
#learning rate & weight

weight = 0.5
input = 0.5
goal_prediction = 0.8

lr = 0.001 #0.1 / 1 / 0.001 / 10 / 100 = optimizer
#10이나 100 주면 거꾸로 줄어든다 ;; 
#activation이 빠져 있긴 하지만 weight 고정인 상태
#한 layer에서 이 역할 

for iteration in range(1101):

    prediction = input * weight
    error = (prediction - goal_prediction) **2 #loss, cost, error

    print("Error: ", str(error) + "\tPrediction: "+str(prediction))

    up_prediction = input * (weight + lr)
    up_error = (goal_prediction - up_prediction)**2

    down_prediction = input * (weight - lr)
    down_error = (goal_prediction - down_prediction)**2

    if(down_error < up_error):
        weight = weight - lr

    if(down_error > up_error):
        weight = weight + lr







