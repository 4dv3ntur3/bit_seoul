#2020-11-23 (11일차)
#0.23.0에서는 돌아가고 0.23.1에는 돌아가지 않는 소스 
#cmd -> pip list 하면 설치된 라이브러리와 버전 확인 가능
#regressor에 있는 모델 추출 

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.utils.testing import all_estimators #testing 빼도 돌아감
import warnings

warnings.filterwarnings('ignore')
iris = pd.read_csv('./data/csv/boston_house_prices.csv', header=1, index_col=0)

x = iris.iloc[:, :-1]
y = iris.iloc[:, -1:]




x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=44
)

allAlgorithms = all_estimators(type_filter='regressor') #이걸 지원을 안 함


for (name, algorithm) in allAlgorithms: 
    #다운그레이드하지 않고 이 방법을 쓰면 새로운 버전? 패치돼서 볼 수 없는 버전?은 나오지 않는다
    try:
        model = algorithm() #모든 모델의 classifier 알고리즘 
                            #알고리즘 하나가 지원을 안 하는 것 
                            #try, catch 사용해서 소스 완성하기 (에러 건너뛰기)

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(name, '의 정답률: ', r2_score(y_test, y_pred))

    except:
        print(name, "은 없는 놈!") #이러면 걸려서 출력되지 않는 것들은 이름이라도 알 수 있다
        # pass #혹은 continue

import sklearn
print(sklearn.__version__) # 0.22.1 버전에 문제가 있어서 출력이 안 됨 -> 버전 낮춰야 함 


#2020-11-23
#몹시 중요한 날!
#차원 압축(PCA): 속도 상승
#feature importance
#validation 왜 한 번만 하나요? 

'''
맹신은 x

ARDRegression 의 정답률:  0.7413660842741397
AdaBoostRegressor 의 정답률:  0.8487892805378938
BaggingRegressor 의 정답률:  0.8911014910820059
BayesianRidge 의 정답률:  0.7397243134288036   
CCA 의 정답률:  0.7145358120880194
DecisionTreeRegressor 의 정답률:  0.8449805619194048
DummyRegressor 의 정답률:  -0.0007982049217318821
ElasticNet 의 정답률:  0.6952835513419808        
ElasticNetCV 의 정답률:  0.6863712064842076
ExtraTreeRegressor 의 정답률:  0.6719927506662602
ExtraTreesRegressor 의 정답률:  0.8932184881087268
GammaRegressor 의 정답률:  -0.0007982049217318821 
GaussianProcessRegressor 의 정답률:  -5.586473869478007
GeneralizedLinearRegressor 의 정답률:  0.6899090511022785
GradientBoostingRegressor 의 정답률:  0.8991078384536849
HistGradientBoostingRegressor 의 정답률:  0.8843141840898427
HuberRegressor 의 정답률:  0.7650865977198575
KNeighborsRegressor 의 정답률:  0.6550811467209019
KernelRidge 의 정답률:  0.7635967086119403
Lars 의 정답률:  0.7440140846099281
LarsCV 의 정답률:  0.7499770153318335
Lasso 의 정답률:  0.683233856987759
LassoCV 의 정답률:  0.7121285098074346
LassoLars 의 정답률:  -0.0007982049217318821
LassoLarsCV 의 정답률:  0.7477692079348518
LassoLarsIC 의 정답률:  0.74479154708417
LinearRegression 의 정답률:  0.7444253077310314
LinearSVR 의 정답률:  0.6037289951944015
MLPRegressor 의 정답률:  0.5231660721586388
MultiTaskElasticNet 의 정답률:  0.6952835513419808
MultiTaskElasticNetCV 의 정답률:  0.6863712064842077
MultiTaskLasso 의 정답률:  0.6832338569877592
MultiTaskLassoCV 의 정답률:  0.7121285098074348
NuSVR 의 정답률:  0.32492104048309933
OrthogonalMatchingPursuit 의 정답률:  0.5661769106723642
OrthogonalMatchingPursuitCV 의 정답률:  0.7377665753906504
PLSCanonical 의 정답률:  -1.3005198325202088
PLSRegression 의 정답률:  0.7600229995900802
PassiveAggressiveRegressor 의 정답률:  0.024719801300767008
PoissonRegressor 의 정답률:  0.79037942981536
RANSACRegressor 의 정답률:  0.6084201906264518
RandomForestRegressor 의 정답률:  0.8867304710341853
Ridge 의 정답률:  0.7465337048988421
RidgeCV 의 정답률:  0.7452747021926976
SGDRegressor 의 정답률:  -3.309618107650224e+26
SVR 의 정답률:  0.2867592174963418
TheilSenRegressor 의 정답률:  0.7794672706913877
TransformedTargetRegressor 의 정답률:  0.7444253077310314
TweedieRegressor 의 정답률:  0.6899090511022785
0.23.1

*******8Ridge & Lasso 
'''