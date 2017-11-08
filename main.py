import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import train


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


# hyper-parameters config
MAX_EPISODES = 1000000
SEQ_LEN = 5
INPUT_DIM = 1
HIDDEN_SIZE = 128
BATCH_SIZE = 128
LEARNING_RATE = 0.001

# main code
trainer = train.Trainer(SEQ_LEN, INPUT_DIM, HIDDEN_SIZE, BATCH_SIZE, LEARNING_RATE)
for i in range(MAX_EPISODES):
	# input_batch = generate_sanity_check_batch(N, BATCH_SIZE)
	input_batch = generate_random_batch(N, BATCH_SIZE)
	correct_out = generate_sorted_onehot(input_batch)

	trainer.train(input_batch, correct_out)

	if i % 1000 == 0:
		trainer.save_model(i)
