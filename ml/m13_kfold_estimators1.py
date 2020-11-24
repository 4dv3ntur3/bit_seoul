#2020-11-24 (12일차)
#cross validation(cv): k-fold 모델별 비교 
#분류

import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators #testing 빼도 돌아감
import warnings

warnings.filterwarnings('ignore')
iris = pd.read_csv('./data/csv/iris_ys.csv', header=0, index_col=0)

x = iris.iloc[:, 0:4]
y = iris.iloc[:, 4]

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=44
)


# allAlgorithms = all_estimators(type_filter='regressor') #이렇게 하면 죄다 nan 나옴
allAlgorithms = all_estimators(type_filter='classifier') 


for (name, algorithm) in allAlgorithms: 
    #다운그레이드하지 않고 이 방법을 쓰면 새로운 버전? 패치돼서 볼 수 없는 버전?은 나오지 않는다
    try:
        model = algorithm() #모든 모델의 classifier 알고리즘 
                            #알고리즘 하나가 지원을 안 하는 것 

        kfold = KFold(n_splits=5, shuffle=True) #몇 개로 조각낼 것인지(n_splits)

        # model.fit(x_train, y_train)
        # y_pred = model.predict(x_test)
        # print(name, '의 정답률: ', accuracy_score(y_test, y_pred))
        scores = cross_val_score(model, x_train, y_train, cv=kfold) #검증한 score
        print(model, ": ", scores)

    except:
        print(name, "은 없는 놈!") #이러면 걸려서 출력되지 않는 것들은 이름이라도 알 수 있다
        # pass #혹은 continue

import sklearn
print(sklearn.__version__) # 0.22.1 버전에 문제가 있어서 출력이 안 됨 -> 버전 낮춰야 함 


'''
n_splits=5
AdaBoostClassifier() :  [0.95833333 0.95833333 0.91666667 0.95833333 0.875     ]
BaggingClassifier() :  [0.91666667 1.         0.91666667 0.95833333 0.95833333]
BernoulliNB() :  [0.20833333 0.33333333 0.29166667 0.29166667 0.375     ]
CalibratedClassifierCV() :  [0.83333333 0.875      0.83333333 0.95833333 0.83333333]
CategoricalNB() :  [0.95833333 0.875      0.875      0.95833333 1.        ]
CheckingClassifier() :  [0. 0. 0. 0. 0.]
ClassifierChain 은 없는 놈!
ComplementNB() :  [0.625      0.625      0.625      0.79166667 0.625     ]
DecisionTreeClassifier() :  [0.95833333 0.95833333 0.875      1.         0.95833333]
DummyClassifier() :  [0.5        0.375      0.20833333 0.29166667 0.41666667]
ExtraTreeClassifier() :  [0.91666667 0.91666667 0.95833333 0.95833333 0.95833333]
ExtraTreesClassifier() :  [0.95833333 1.         0.95833333 0.875      0.95833333]
GaussianNB() :  [1.         0.95833333 0.95833333 0.91666667 0.95833333]
GaussianProcessClassifier() :  [0.95833333 1.         0.91666667 0.91666667 1.        ]
GradientBoostingClassifier() :  [0.95833333 0.875      0.95833333 0.91666667 0.95833333]
HistGradientBoostingClassifier() :  [0.875      0.91666667 1.         0.91666667 0.91666667]
KNeighborsClassifier() :  [0.95833333 0.875      0.95833333 0.91666667 1.        ]
LabelPropagation() :  [0.91666667 1.         1.         0.875      0.95833333]
LabelSpreading() :  [0.91666667 0.91666667 1.         0.95833333 1.        ]
LinearDiscriminantAnalysis() :  [1.         0.95833333 1.         0.95833333 1.        ]
LinearSVC() :  [0.91666667 0.91666667 1.         0.91666667 0.875     ]
LogisticRegression() :  [0.95833333 0.95833333 0.875      0.91666667 0.95833333]
LogisticRegressionCV() :  [0.95833333 0.95833333 0.95833333 0.95833333 0.91666667]
MLPClassifier() :  [1.         0.95833333 0.95833333 1.         0.91666667]
MultiOutputClassifier 은 없는 놈!
MultinomialNB() :  [0.79166667 0.95833333 0.91666667 0.66666667 0.5       ]
NearestCentroid() :  [0.91666667 0.91666667 0.95833333 1.         0.83333333]
NuSVC() :  [0.91666667 1.         0.95833333 0.95833333 0.95833333]
OneVsOneClassifier 은 없는 놈!
OneVsRestClassifier 은 없는 놈!
OutputCodeClassifier 은 없는 놈!
PassiveAggressiveClassifier() :  [0.5        0.75       0.83333333 0.83333333 0.625     ]
Perceptron() :  [0.54166667 0.54166667 0.95833333 0.58333333 0.83333333]
QuadraticDiscriminantAnalysis() :  [1.         1.         0.95833333 0.875      1.        ]
RadiusNeighborsClassifier() :  [0.95833333 0.83333333 0.875      0.95833333 1.        ]
RandomForestClassifier() :  [0.95833333 0.95833333 0.875      0.95833333 0.95833333]
RidgeClassifier() :  [0.875      0.875      0.70833333 0.875      0.875     ]
RidgeClassifierCV(alphas=array([ 0.1,  1. , 10. ])) :  [0.875      0.91666667 0.79166667 0.83333333 0.875     ]
SGDClassifier() :  [1.         0.66666667 0.95833333 0.70833333 0.75      ]
SVC() :  [0.875      1.         0.91666667 0.91666667 0.95833333]
StackingClassifier 은 없는 놈!
VotingClassifier 은 없는 놈!
'''