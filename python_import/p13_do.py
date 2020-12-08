#2020-12-08
#python 함수

import p11_car
import p12_tv


#main은 자기 자체 
#다른 코드에서 import하면 그 파일의 파일명이 출력된다 
#그 파일을 실행시키는 놈이 main

print("==============")
print("do.py의 module 이름은 ", __name__)


'''
운전하다
car.py의 module 이름은  p11_car
시청하다
tv.py의 module 이름은  p12_tv
==============
do.py의 module 이름은  __main__
'''


print("================")
p11_car.drive() #함수 밖에 있는 건 안 당겨오고 함수만 가져옴 
p12_tv.watch()

'''
================
운전하다
시청하다
'''