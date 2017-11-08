import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import model

class Trainer:

	def __init__(self, seq_len, input_dim, hidden_dim=256, batch_size=128, learning_rate=0.001):
		self.seq_len = seq_len
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		self.episode = 0

		self.ptrNet = model.PtrNet(self.batch_size, self.seq_len, self.input_dim, self.hidden_dim)

		self.optimizer = torch.optim.RMSprop(self.ptrNet.parameters(), self.learning_rate)

	def incorrect_count_loss(self, actual, pred):
		actual = actual.data.numpy()
		pred = pred.data.numpy()
		sum = 0.0
		predArg = np.argmax(pred, axis=2)
		for i in range(pred.shape[0]):
			for j in range(pred.shape[1]):
				if actual[i][j][predArg[i][j]] != 1:
					sum += 1.0
		return sum/pred.shape[1]

	def train(self, input, ground_truth):
		correct_out = Variable(torch.from_numpy(ground_truth))
		pred_out = self.ptrNet.forward(input)

		loss = torch.sqrt(torch.mean(torch.pow(correct_out - pred_out, 2)))
		loss.backward()
		self.optimizer.step()

		self.episode += 1
		print 'Episode :- ', self.episode, ' L2 Loss :- ', loss.data.numpy(), \
			' My Loss :- ', self.incorrect_count_loss(correct_out, pred_out)

	def test_batch(self, input):
		pred_out = self.ptrNet.forward(input)




	def save_model(self, episode_count):
		"""
		saves the model
		:param episode_count: the count of episodes iterated
		:return:
		"""
		torch.save(self.ptrNet.state_dict(), './Models/' + str(episode_count) + '_net.pt')
		print 'Model saved successfully'

	def load_model(self, episode):
		"""
		loads the model
		:param episode: the count of episodes iterated (used to find the file name)
		:return:
		"""
		self.ptrNet.load_state_dict(torch.load('./Models/' + str(episode) + '_net.pt'))
		print 'Model loaded succesfully'