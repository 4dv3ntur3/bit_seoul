#2020-11-23 (11일차)
#0.23.0에서는 돌아가고 0.23.1에는 돌아가지 않는 소스 
#cmd -> pip list 하면 설치된 라이브러리와 버전 확인 가능
#classifier에 있는 모델 추출 

import pandas as pd
from sklearn.model_selection import train_test_split
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

allAlgorithms = all_estimators(type_filter='classifier') #이걸 지원을 안 함

for (name, algorithm) in allAlgorithms: 
    #다운그레이드하지 않고 이 방법을 쓰면 새로운 버전? 패치돼서 볼 수 없는 버전?은 나오지 않는다
    try:
        model = algorithm() #모든 모델의 classifier 알고리즘 
                            #알고리즘 하나가 지원을 안 하는 것 

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(name, '의 정답률: ', accuracy_score(y_test, y_pred))

    except:
        print(name, "은 없는 놈!") #이러면 걸려서 출력되지 않는 것들은 이름이라도 알 수 있다
        # pass #혹은 continue

import sklearn
print(sklearn.__version__) # 0.22.1 버전에 문제가 있어서 출력이 안 됨 -> 버전 낮춰야 함 


'''
맹신은 x

AdaBoostClassifier 의 정답률:  0.9666666666666667
BaggingClassifier 의 정답률:  0.9666666666666667
BernoulliNB 의 정답률:  0.3
CalibratedClassifierCV 의 정답률:  0.9333333333333333
CategoricalNB 의 정답률:  0.9
CheckingClassifier 의 정답률:  0.3
ComplementNB 의 정답률:  0.7
DecisionTreeClassifier 의 정답률:  0.8666666666666667
DummyClassifier 의 정답률:  0.26666666666666666
ExtraTreeClassifier 의 정답률:  0.9666666666666667
ExtraTreesClassifier 의 정답률:  0.9666666666666667
GaussianNB 의 정답률:  0.9333333333333333
GaussianProcessClassifier 의 정답률:  0.9666666666666667
GradientBoostingClassifier 의 정답률:  0.9666666666666667
HistGradientBoostingClassifier 의 정답률:  0.9666666666666667
KNeighborsClassifier 의 정답률:  0.9666666666666667
LabelPropagation 의 정답률:  0.9666666666666667
LabelSpreading 의 정답률:  0.9666666666666667
LinearDiscriminantAnalysis 의 정답률:  1.0
LinearSVC 의 정답률:  0.9666666666666667
LogisticRegression 의 정답률:  0.9666666666666667
LogisticRegressionCV 의 정답률:  0.9666666666666667
MLPClassifier 의 정답률:  1.0
MultinomialNB 의 정답률:  0.8666666666666667
NearestCentroid 의 정답률:  0.9
NuSVC 의 정답률:  0.9666666666666667
PassiveAggressiveClassifier 의 정답률:  1.0
Perceptron 의 정답률:  0.7333333333333333
QuadraticDiscriminantAnalysis 의 정답률:  1.0
RadiusNeighborsClassifier 의 정답률:  0.9333333333333333
RandomForestClassifier 의 정답률:  0.9666666666666667
RidgeClassifier 의 정답률:  0.8333333333333334
RidgeClassifierCV 의 정답률:  0.8333333333333334
SGDClassifier 의 정답률:  0.6333333333333333
SVC 의 정답률:  0.9666666666666667
0.23.1
'''