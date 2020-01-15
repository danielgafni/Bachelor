from thesis.nets import *
from thesis.utils import *

"""
Run this script to make sure everything is ok.
"""


print('Testing LC_SNN...')

try:
    net = LC_SNN()
except:
    print('Failed to create a LC_SNN')
try:
    net.train(2)
except:
    print('Failed to train the LC_SNN')
try:
    net.collect_activity(2)
except:
    print('Failed to collect activity from the LC_SNN')
try:
    net.calibrate(2)
except:
    print('Failed to calibrate the LC_SNN')
try:
    net.calculate_accuracy(2)
except:
    print('Failed to calculate accuracy of the LC_SNN')
try:
    net.calibrate_lc(2)
except:
    print('Failed to calibrate the linear classifier for the LC_SNN')
try:
    net.calculate_accuracy_lc(2)
except:
    print('Failed to calculate accuracy of the linear classifier for the LC_SNN')
try:
    net.save()
except:
    print('Failed to save the LC_SNN')
try:
    delete_network(net.name, sure=True)
except:
    print('Failed to delete the LC_SNN')

print('Test finished.')