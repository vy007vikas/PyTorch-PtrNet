import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class PtrNet(nn.Module):

	def __init__(self, batch_size, seq_len, input_dim, hidden_dim):
		super(PtrNet, self).__init__()

		self.batch_size = batch_size				# B
		self.seq_len = seq_len						# N
		self.input_dim = input_dim					# I
		self.hidden_dim = hidden_dim				# H

		# encoder
		self.encoder = []
		for i in range(self.seq_len):
			cell = nn.LSTMCell(input_dim, hidden_dim)
			self.encoder.append(cell)

		# decoder
		self.decoder = []
		for i in range(self.seq_len):
			cell = nn.LSTMCell(input_dim, hidden_dim)
			self.decoder.append(cell)

		# for creating pointers
		self.W_encoder = nn.Linear(self.hidden_dim, self.hidden_dim)
		self.W_decoder = nn.Linear(self.hidden_dim, self.hidden_dim)
		self.V = nn.Linear(self.hidden_dim, self.input_dim)

	def forward(self, input):
		encoded_input = []

		# initialize hidden state and cell state as random
		h = Variable(torch.randn([self.batch_size, self.hidden_size]))		# B*H
		c = Variable(torch.randn([self.batch_size, self.hidden_size]))		# B*H
		for i in range(self.seq_len):
			h, c = self.encoder[i](input[i], (h, c))						# B*H
			encoded_input.append(c)

		d_i = Variable(torch.Tensor([-1]*self.batch_size).view(self.batch_size, self.seq_len))			# B*I
		distributions = []
		for i in range(self.seq_len):
			h, c = self.decoder[i](d_i, (h, c))				# B*H

			# the attention part as obtained from the paper
			# u_i[j] = v * tanh(W1 * e[j] + W2 * c_i)
			u_i = []
			c_i = self.W_decoder(c)								# B*H
			for j in range(self.seq_len):
				e_j = self.W_encoder(encoded_input[j])			# B*H
				u_j = self.V(F.tanh(c_i + e_j)).squeeze(1)		# B*I
				u_i.append(u_j)

			# a_i[j] = softmax(u_i[j])
			u_i = torch.stack(u_i).t()			# N*B -> B*N
			a_i = F.softmax(u_i)				# B*N
			distributions.append(a_i)

			# d_i+1 = sum(a_i[j]*e[j]) over j
			d_i = 0
			for j in range(self.seq_len):
				# select jth column of a_i
				a_j = torch.index_select(a_i, 1, torch.LongTensor([j]))			# B,
				a_j = torch.expand(self.batch_size, self.hidden_dim)			# B*H
				d_i = d_i + (a_j*encoded_input[j])								# B*H

		return distributions					# N*B*N
