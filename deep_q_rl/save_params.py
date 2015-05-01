import cPickle
import sys
import numpy 
import os

file_name = sys.argv[1]

directoryParamSaving = '/home/ubuntu/project/bayesian/parametersAmazon/breakout/dqn'

if not os.path.exists(directoryParamSaving):
	os.makedirs(directoryParamSaving)	

fileHiddenParams = directoryParamSaving+'/hidden.npz'
fileLogisticParams = directoryParamSaving+'/logistic.npz'
fileConvolutional1 = directoryParamSaving+'/conv1.npz'
fileConvolutional2 = directoryParamSaving+'/conv2.npz'

net_file = open(file_name, 'r')
net = cPickle.load(net_file)

if net.approximator == 'conv':
	first = 1
else:
	first = 2

W = net.q_layers[first].W.get_value()
b = net.q_layers[first].b.get_value()
numpy.savez(fileConvolutional1, W, b ) 
print W.shape
print b.shape

W = net.q_layers[first+1].W.get_value()
b = net.q_layers[first+1].b.get_value()
numpy.savez(fileConvolutional2, W, b )

print W.shape

if net.approximator == 'conv':
	first = 3
else:
	first = 5

W = net.q_layers[first].W.get_value()
b = net.q_layers[first].b.get_value()
numpy.savez(fileHiddenParams, W, b )



W = net.q_layers[first+1].W.get_value()
b = net.q_layers[first+1].b.get_value()

print W.shape
print b.shape
numpy.savez(fileLogisticParams, W, b )

