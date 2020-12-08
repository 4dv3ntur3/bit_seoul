#2020-12-08

#path가 걸려 있는 Anaconda3 폴더 (어디서든 불러올 수 있다)
from test_1208 import p62_import 

p62_import.sum2()


print("=============================")


from test_1208.p62_import import sum2
sum2()



'''
이 파일은 아나콘다 폴더에 들어 있을 것이다! #print문 출력됨
작업그룹 import 썸탄다!!! #sum2
=============================
작업그룹 import 썸탄다!!! #sum2
'''