import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import model

MAX_EPISODES = 1000000
N = 5
BATCH_SIZE = 128
LEARNING_RATE = 0.001


def generate_sanity_check_batch(len, batch_size=64):
	out = np.arange(len).reshape((1,len))
	for i in range(batch_size-1):
		out = np.append(out, np.arange(len).reshape((1,len)), axis=0)
	return out


def generate_random_batch(len, batch_size=64):
	return np.random.rand(batch_size, len)


def generate_sorted_onehot(input):
	input = np.array(input, dtype=np.float32)
	out = np.zeros((input.shape[1], input.shape[0], input.shape[1]), dtype=np.float32)
	for a in range(input.shape[0]):
		ind = [b[0] for b in sorted(enumerate(input[a]), key=lambda i:i[1])]
		for j in range(len(ind)):
			out[j][a][ind[j]] = 1
	return out


def sanity_check_sorted_onehot():
	arr = [[3,1,2],[1,2,3]]
	out = generate_sorted_onehot(arr)
	print out


def my_loss(actual, pred):
	actual = actual.data.numpy()
	pred = pred.data.numpy()
	sum = 0.0
	predArg = np.argmax(pred, axis=2)
	for i in range(pred.shape[0]):
		for j in range(pred.shape[1]):
			if actual[i][j][predArg[i][j]] != 1:
				sum += 1.0
	return sum/pred.shape[1]


def adjust_learning_rate(lr, optimizer, epoch):
	lr = lr * (0.96 ** (epoch // 30))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr
	return lr


# main code
ptrNet = model.PtrNet(BATCH_SIZE, N, 1, 128)
optimizer = torch.optim.RMSprop(ptrNet.parameters(), LEARNING_RATE)
for i in range(MAX_EPISODES):
	# input_batch = generate_sanity_check_batch(N, BATCH_SIZE)
	input_batch = generate_random_batch(N, BATCH_SIZE)
	correct_out = generate_sorted_onehot(input_batch)

	correct_out = Variable(torch.from_numpy(correct_out))
	pred_out = ptrNet.forward(input_batch)

	loss = torch.sqrt(torch.mean(torch.pow(correct_out - pred_out, 2)))
	loss.backward()
	optimizer.step()

	print 'Episode :- ', i, ' L2 Loss :- ', loss.data.numpy(), ' My Loss :- ', my_loss(correct_out, pred_out), " LR :- ", LEARNING_RATE
