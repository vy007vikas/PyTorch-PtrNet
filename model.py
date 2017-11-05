import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class PtrNet(nn.Module):

	def __init__(self, batch_size, seq_len, input_dim, hidden_dim, hidden_len):
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

		# for pointers
		self.W_encoder = nn.Linear(self.hidden_dim, self.hidden_dim)
		self.W_decoder = nn.Linear(self.hidden_dim, self.hidden_dim)
		self.V = nn.Linear(self.hidden_dim, self.hidden_dim)


	def forward(self, input):
		encoded_input = []

		# initialize hidden state and cell state as random
		h = Variable(torch.randn([self.batch_size, self.hidden_size]))		# B*H
		c = Variable(torch.randn([self.batch_size, self.hidden_size]))		# B*H
		for i in range(self.seq_len):
			h, c = self.encoder[i](input[i], (h, c))						# B*H
			encoded_input.append(c)

		d_i = Variable(torch.Tensor([-1]*self.batch_size).view(self.batch_size, self.seq_len))			# B*I
		for i in range(self.seq_len):
			h, c = self.decoder[i](d_i, (h, c))				# B*H

			# the attention part as obtained from the paper
			# u_i[j] = v * tanh(W1 * e[j] + W2 * c_i)
			c1 = self.W_decoder(c)							# B*H
			for j in range(self.seq_len):
				e1 = self.W_encoder(encoded_input[j])		# B*H
				u = self.V(nn.tanh(c1 + e1))				# B*H
				














