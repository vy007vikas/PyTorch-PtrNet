import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class PtrNet(nn.Module):

	def __init__(self, batch_size, seq_len, input_dim, hidden_dim, hidden_len):
		super(PtrNet, self).__init__()

		self.batch_size = batch_size				# B

		# encoder
		self.seq_len = seq_len						# N
		self.input_dim = input_dim					# I
		self.hidden_dim = hidden_dim				# H

		self.encoder = []
		for i in range(self.seq_len):
			cell = nn.LSTMCell(input_dim, hidden_dim)
			self.encoder.append(cell)

		# decoder

		self.decoder = []
		for i in range(self.seq_len):
			cell = nn.LSTMCell()
			self.decoder.append(cell)

	def forward(self, input):
		encoded_input = []

		# initialize hidden state and cell state as random
		h = Variable(torch.randn([self.batch_size, self.hidden_size]))		# B*H
		c = Variable(torch.randn([self.batch_size, self.hidden_size]))		# B*H
		for i in range(self.seq_len):
			h,c = self.encoder[i](input[i], (h,c))		# B*H
			encoded_input.append(c)









