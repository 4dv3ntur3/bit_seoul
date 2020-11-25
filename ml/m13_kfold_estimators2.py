#2020-11-24 (12일차)
#cross validation(cv): k-fold 모델별 비교 
#회귀

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score
from sklearn.utils.testing import all_estimators #testing 빼도 돌아감



import warnings
warnings.filterwarnings('ignore')


boston = pd.read_csv('./data/csv/boston_house_prices.csv', header=1, index_col=0)

x = boston.iloc[:, :-1]
y = boston.iloc[:, -1:]


x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=44
)

allAlgorithms = all_estimators(type_filter='regressor') #이걸 지원을 안 함
# allAlgorithms = all_estimators(type_filter='classifier') -> 이렇게 하면 죄다 nan 뜬다 


for (name, algorithm) in allAlgorithms: 
    #다운그레이드하지 않고 이 방법을 쓰면 새로운 버전? 패치돼서 볼 수 없는 버전?은 나오지 않는다
    try:
        model = algorithm() #모든 모델의 classifier 알고리즘 
                            #알고리즘 하나가 지원을 안 하는 것 
                            #try, catch 사용해서 소스 완성하기 (에러 건너뛰기)
        kfold = KFold(n_splits=10, shuffle=True) #몇 개로 조각낼 것인지(n_splits)

        # model.fit(x_train, y_train)
        # y_pred = model.predict(x_test)
        # print(name, '의 정답률: ', r2_score(y_test, y_pred))
        scores = cross_val_score(model, x_train, y_train, cv=kfold) #검증한 score
        print(model, ": ", scores)

    except:
        print(name, "은 없는 놈!") #이러면 걸려서 출력되지 않는 것들은 이름이라도 알 수 있다
        # pass #혹은 continue

import sklearn
print(sklearn.__version__) # 0.22.1 버전에 문제가 있어서 출력이 안 됨 -> 버전 낮춰야 함 


'''
n_splits=5
ARDRegression() :  [0.6787983  0.74556119 0.58419812 0.73066572 0.69197472]
AdaBoostRegressor() :  [0.91092986 0.84932723 0.65831556 0.83653246 0.8899729 ]
BaggingRegressor() :  [0.84679246 0.87829648 0.84617765 0.80877412 0.87118048]
BayesianRidge() :  [0.69754539 0.64980008 0.75917518 0.72854576 0.63545546]
CCA() :  [0.69824786 0.46984917 0.47632311 0.72535707 0.77530933]
DecisionTreeRegressor() :  [0.76175306 0.84339868 0.47548031 0.80969668 0.66253798]
DummyRegressor() :  [-0.00177602 -0.0278794  -0.03250944 -0.0435377  -0.0274448 ]
ElasticNet() :  [0.70350243 0.69569545 0.52757518 0.73468913 0.60346771]
ElasticNetCV() :  [0.6694736  0.68482841 0.63870725 0.5556229  0.67050703]
ExtraTreeRegressor() :  [0.65456934 0.49213724 0.82884108 0.65187183 0.84613132]
ExtraTreesRegressor() :  [0.84998813 0.87175715 0.8745022  0.89579452 0.89971192]
GammaRegressor 은 없는 놈!
GaussianProcessRegressor() :  [-7.5081997  -5.23874502 -6.10190034 -5.61506694 -6.12972671]
GeneralizedLinearRegressor 은 없는 놈!
GradientBoostingRegressor() :  [0.91597345 0.79920018 0.91316643 0.8618669  0.82264107]
HistGradientBoostingRegressor() :  [0.88516062 0.72068631 0.87912611 0.82305232 0.89485399]
HuberRegressor() :  [0.59899029 0.63774168 0.55000585 0.67730454 0.76867208]
IsotonicRegression() :  [nan nan nan nan nan]
KNeighborsRegressor() :  [0.48603586 0.6377431  0.30736867 0.34928593 0.38044917]
KernelRidge() :  [0.7619673  0.58126928 0.70973599 0.48476733 0.75208632]
Lars() :  [0.73667839 0.71527943 0.7184914  0.57971832 0.64985786]
LarsCV() :  [0.69419142 0.68114268 0.72901654 0.76235914 0.67358092]
Lasso() :  [0.6567932  0.62620255 0.64683784 0.59647099 0.65078993]
LassoCV() :  [0.71346763 0.46931726 0.74614036 0.71816272 0.61606213]
LassoLars() :  [-0.00416524 -0.07466426 -0.00400696 -0.02112667 -0.01934369]
LassoLarsCV() :  [0.73702722 0.78820568 0.67270908 0.62570283 0.64756864]
LassoLarsIC() :  [0.58479173 0.77006929 0.6772283  0.66157358 0.66943919]
LinearRegression() :  [0.72720642 0.74965282 0.73113143 0.58069078 0.73529933]
LinearSVR() :  [-3.53998061  0.45408816  0.62433253  0.62030004  0.62999502]
MLPRegressor() :  [0.66070661 0.47298325 0.60305913 0.58422108 0.50663614]
MultiOutputRegressor 은 없는 놈!
MultiTaskElasticNet() :  [0.52351053 0.60132203 0.69240608 0.71919883 0.5693822 ]
MultiTaskElasticNetCV() :  [0.69562065 0.63749479 0.62756666 0.6317022  0.5652854 ]
MultiTaskLasso() :  [0.68888043 0.66272112 0.57447139 0.61635165 0.67010307]
MultiTaskLassoCV() :  [0.60425773 0.617501   0.6545622  0.69784784 0.69852393]
NuSVR() :  [ 0.30850409  0.33388219  0.21781448  0.17744546 -0.01298135]
OrthogonalMatchingPursuit() :  [0.62689178 0.54892979 0.43879058 0.4729687  0.5918662 ]
OrthogonalMatchingPursuitCV() :  [0.5500725  0.59756471 0.6976935  0.77904519 0.64687714]
PLSCanonical() :  [-1.50219624 -1.63310323 -2.98948905 -2.53824543 -1.25487705]
PLSRegression() :  [0.70152085 0.75721767 0.77670268 0.49596664 0.60439603]
PassiveAggressiveRegressor() :  [-1.4849862  -1.48455427  0.16979922 -0.1248793  -1.85475974]
PoissonRegressor 은 없는 놈!
RANSACRegressor() :  [0.78485605 0.60297075 0.57175568 0.32251983 0.19864172]
RadiusNeighborsRegressor 은 없는 놈!
RandomForestRegressor() :  [0.80524391 0.87066396 0.88662404 0.84838618 0.90572494]
RegressorChain 은 없는 놈!
Ridge() :  [0.76495027 0.59790324 0.68102547 0.56073711 0.80692798]
RidgeCV(alphas=array([ 0.1,  1. , 10. ])) :  [0.70171233 0.72690215 0.62889946 0.71816198 0.68988223]
SGDRegressor() :  [-7.27476013e+25 -7.03263971e+24 -1.16324063e+26 -1.50566575e+26
 -2.27785334e+26]
SVR() :  [ 0.30320333  0.31688198 -0.03208049  0.12633198  0.1707922 ]
StackingRegressor 은 없는 놈!
TheilSenRegressor(max_subpopulation=10000) :  [0.58418161 0.75132071 0.5660568  0.73868928 0.67288846]
TransformedTargetRegressor() :  [0.62080673 0.82453979 0.69478989 0.7566284  0.59685291]
TweedieRegressor 은 없는 놈!
VotingRegressor 은 없는 놈!
_SigmoidCalibration() :  [nan nan nan nan nan]
'''