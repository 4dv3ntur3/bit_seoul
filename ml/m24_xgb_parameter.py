#2020-11-25 (13일차)
#과적합 방지
#1. 훈련데이터량을 늘린다
#2. 피쳐수를 줄인다
#3. regularization


#다한 사람은 모델을 완성해서 결과 주석으로 적어놓을 것 
from xgboost import XGBClassifier, XGBRegressor, plot_importance
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split




n_estimators = 300
learning_rate = 1
colsample_bytree = 1
colsample_bylevel = 1

max_depth = 5
n_jobs = -1 #CPU를 다 쓸 것인지 말 것인지


#score로 성능 비교

datasets = load_boston()

model = XGBRegressor(
    max_depth=max_depth, 
    learning_rate=learning_rate, n_estimators=n_estimators, 
    n_jobs=n_jobs, 
    colsample_bylevel = colsample_bylevel, 
    colsample_bytree=colsample_bytree)


x_train, x_test, y_train, y_test = train_test_split(
    datasets.data, datasets.target, random_state=66, shuffle=True, train_size=0.8 
)


scaler = StandardScaler()
scaler.fit(x_train)

scaler.transform(x_train)
scaler.transform(x_test)


model.fit(x_train, y_train)

from sklearn.metrics import r2_score

score = model.score(x_test, y_test)     #회귀에서는 r2_score = score이다
                                        #accuracy_score != score (둘이 같다는 건 분류에서나 통하는 이야기)
print("model.score: ", score)



'''
n_estimators = 300
learning_rate = 1
colsample_bytree = 1
colsample_bylevel = 1

max_depth = 5
n_jobs = -1 #CPU를 다 쓸 것인지 말 것인지

model.score:  0.8454028763099724
#제대로 된 튠도 아닌 상황에서 근소한 차이. 무엇보다 속도가 매우 빠르다!

'''



'''
# General Parameter
    booster='gbtree' # 트리,회귀(gblinear) 트리가 항상 
                     # 더 좋은 성능을 내기 때문에 수정할 필요없다고한다.
    
    silent=True  # running message출력안한다.
                 # 모델이 적합되는 과정을 이해하기위해선 False으로한다.
    
    min_child_weight=10   # 값이 높아지면 under-fitting 되는 
                          # 경우가 있다. CV를 통해 튜닝되어야 한다.
    
    max_depth=8     # 트리의 최대 깊이를 정의함. 
                    # 루트에서 가장 긴 노드의 거리.
                    # 8이면 중요변수에서 결론까지 변수가 9개거친다.
                    # Typical Value는 3-10. 
    
    gamma =0    # 노드가 split 되기 위한 loss function의 값이
                # 감소하는 최소값을 정의한다. gamma 값이 높아질 수록 
                # 알고리즘은 보수적으로 변하고, loss function의 정의
                #에 따라 적정값이 달라지기때문에 반드시 튜닝.
    
    nthread =4    # XGBoost를 실행하기 위한 병렬처리(쓰레드)
                  #갯수. 'n_jobs' 를 사용해라.
    
    colsample_bytree=0.8   # 트리를 생성할때 훈련 데이터에서 
                           # 변수를 샘플링해주는 비율. 보통0.6~0.9
    
    colsample_bylevel=0.9  # 트리의 레벨별로 훈련 데이터의 
                           #변수를 샘플링해주는 비율. 보통0.6~0.9
    
    n_estimators =(int)   #부스트트리의 양
                          # 트리의 갯수. 
    
    objective = 'reg:linear','binary:logistic','multi:softmax',
                'multi:softprob'  # 4가지 존재.
            # 회귀 경우 'reg', binary분류의 경우 'binary',
            # 다중분류경우 'multi'- 분류된 class를 return하는 경우 'softmax'
            # 각 class에 속할 확률을 return하는 경우 'softprob'
    
    random_state =  # random number seed.
                    # seed 와 동일.
)




XGBClassifier.fit(
    
    X (array_like)     # Feature matrix ( 독립변수)
                       # X_train
    
    Y (array)          # Labels (종속변수)
                       # Y_train
    
    eval_set           # 빨리 끝나기 위해 검증데이터와 같이써야한다.  
                       # =[(X_train,Y_train),(X_vld, Y_vld)]
 
    eval_metric = 'rmse','error','mae','logloss','merror',
                'mlogloss','auc'  
              # validation set (검증데이터)에 적용되는 모델 선택 기준.
              # 평가측정. 
              # 회귀 경우 rmse ,  분류 -error   이외의 옵션은 함수정의
    
    early_stopping_rounds=100,20
              # 100번,20번 반복동안 최대화 되지 않으면 stop
)






'''
import matplotlib.pyplot as plt


plot_importance(model) #XGBooster에서 제공해 줌
                       #f score 확인해 볼 것. feature importance의 평가 지표
                       #customizing도 가능하긴 하다
plt.show()


# #feature importance
# import numpy as np
# def plot_feature_importances_cancer(model):
#     n_features = datasets.data.shape[1]
#     plt.barh(np.arange(n_features), model.feature_importances_,
#                 align='center')

#     plt.yticks(np.arange(n_features), datasets.feature_names)
#     plt.xlabel("feature importances")
#     plt.ylabel("features")
#     plt.ylim(-1, n_features)

# plot_feature_importances_cancer(model)
# plt.show()
