#2020-12-08

#배포해서 하는 방식도 있는데 그렇게 하면 어디서든 당겨 올 수 있다 

from machine.car import drive
from machine.tv import watch

drive()
watch()

print("=========================")

from machine import car
from machine import tv

car.drive()
tv.watch()


print("=========================")

from machine.test.car import drive
from machine.test.tv import watch

drive()
watch()

'''
=========================
test 운전하다
test 시청하다
'''

print("=========================")


from machine.test import car
from machine.test import tv

car.drive()
tv.watch()

'''
=========================
test 운전하다
test 시청하다
'''

from machine import test

test.car.drive()
test.tv.watch()


'''
=========================
test 운전하다
test 시청하다
test 운전하다
test 시청하다
'''