import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import model

MAX_EPISODES = 1000
N = 5
BATCH_SIZE = 128


def generate_batch(len, batch_size=64):
	return np.random.rand(batch_size, len)


def generate_sorted_onehot(input):
	out = np.zeros((input.shape[1], input.shape[0], input.shape[1]))
	for a in range(input.shape[0]):
		ind = [b[0] for b in sorted(enumerate(input[a]), key=lambda i:i[1])]
		for j in range(len(ind)):
			out[j][i][ind[j]] = 1
	return out


arr = [[3,1,2],[1,2,3]]
out = generate_sorted_onehot(arr)
print arr

# for i in range(MAX_EPISODES):
# 	input_batch = generate_batch(N, BATCH_SIZE)
# 	correct_out = generate_sorted_onehot(input_batch)
