#2020-11-23 (11일차)
#wine.csv Y값(class) 분포 확인

import pandas as pd


wine = pd.read_csv('./data/csv/winequality-white.csv', sep=';', header=0)

#중요하다(분류모델 할 때 유용할 것)
count_data = wine.groupby('quality')['quality'].count() #quality라는 column을 group화 후 각 group의 개체 수 계산
print(count_data)

# quality
# 3      20
# 4     163
# 5    1457
# 6    2198
# 7     880
# 8     175
# 9       5

#분포 확인
import matplotlib.pyplot as plt
count_data.plot()
plt.show() #0, 1은 아예 배제. 3부터 시작, 5, 6, 7에 치중. 그러면 끝에 있는 data는 쪽수로 밀림
           #데이터가 개/고양이로 구분하는 모델인데 개 900만장, 고양이 10장. -> 그럼 99.9퍼센트 개로 본다
           #고양이 데이터가 적으니까 일단 뭐가 들어오든 개라고 대답하면 90퍼센트는 맞는 것임

           #라벨링 필요! (분포 구간을 짧게) 3~4 하나, 5, 6, 7 하나...


