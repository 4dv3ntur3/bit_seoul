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