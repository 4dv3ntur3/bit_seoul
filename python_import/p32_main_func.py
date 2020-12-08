#2020-12-08
#python 함수

import p31_sample

x = 222 #p31_sample의 x보다 이 x가 더 우선권 있음
def main_func():
    print('x: ', x)

main_func()

'''
x:  222
'''

p31_sample.test() #x:  111