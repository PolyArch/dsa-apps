import sys
sys.path.insert(0,'/usr/local/caffe/python/')
import caffe
import numpy as np
import os

os.system("rm *.txt")

#load the model
net = caffe.Net('/home/vidushi/SkimCaffe/models/bvlc_alexnet/deploy.prototxt',
                '/home/vidushi/SkimCaffe/models/bvlc_alexnet/bvlc_alexnet.caffemodel',
                caffe.TEST)


#load the data(conv and fc layer pretrained weights) in the model -- and check if they are sparse?
layer_names = ['conv2']
for layer in layer_names:
	W = net.params[layer][0].data[...]
	b = net.params[layer][1].data[...]

print("number dim of weights:")
print(np.ndim(W))
print(np.shape(W))
index = np.array(np.shape(W))

#decide the format here
# f = open('weight_file.txt','ab')
# for i in range(index[0]):
# 	for j in range(index[1]):
# 		for k in range(index[2]):
# 			temp = np.array(W[i][j][k])
# 			np.savetxt(f, temp, delimiter=' ')
# 			# for l in range(index[3]):
# 			#	x=0
# f.close()
# np.savetxt('weight_file', W, delimiter=' ')

# load input and configure preprocessing
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', np.load('python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
transformer.set_transpose('data', (2,0,1))
transformer.set_channel_swap('data', (2,1,0))
transformer.set_raw_scale('data', 255.0)

#note we can change the batch size on-the-fly
#since we classify only one image, we change batch size from 10 to 1
net.blobs['data'].reshape(1,3,227,227)

#load the image in the data layer
im = caffe.io.load_image('examples/images/cat.jpg')
net.blobs['data'].data[...] = transformer.preprocess('data', im)

#TODO: print the intermediate feature maps from the model -- and store them in different files (conv1 o/p should be good enough)
out = net.forward()
conv2 = net.blobs['conv1'].data
fc7 = net.blobs['fc7'].data

#act_file = TemporaryFile()
print('act dim: ', np.ndim(conv2))
act_index = np.array(np.shape(conv2))
print(np.shape(conv2))

# np.savetxt('act_file', conv2, delimiter=' ')
#decide the format here
# conv2 = np.around(conv2, decimals=2)
f = open('act_file.txt','ab')
# new_arr = conv2.flatten(order='C')
# print(np.shape(new_arr))

# need to ignore the first dimension which is 1 by default
for j in range(act_index[1]):
	# temp = np.around(conv2[0][j].flatten(order='C'), decimals=2)
	temp = conv2[0][j].flatten(order='C')
	np.savetxt(f, temp, fmt = '%0.2f', delimiter=' ')

# for i in range(act_index[0]):
# 	for j in range(act_index[1]):
# 		for k in range(act_index[2]):
# 			temp = np.around(np.array(conv2[i][j][k]), decimals=2)
# 			print(conv2[i][j][k][0], temp[0])
# 			np.savetxt(f, temp, delimiter=',')
# f.close()

#compute
#out = net.forward()

# other possibility : out = net.forward_all(data=np.asarray([transformer.preprocess('data', im)]))

#predicted predicted class
print(out['prob'].argmax())

#print predicted labels
labels = np.loadtxt("/home/vidushi/caffe/data/ilsvrc12/synset_words.txt", str, delimiter='\t')
top_k = net.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]
print(labels[top_k])
