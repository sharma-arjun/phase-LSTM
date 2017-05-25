import math
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


class LSTM(nn.Module):
	def __init__(self, input_size, output_size, hidden_size, n_layers=1, batch_size=1):
		super(LSTM, self).__init__()

		self.input_size = input_size
		self.output_size = output_size
		self.hidden_size = hidden_size
		self.n_layers = n_layers
		self.batch_size = batch_size


		self.gru1 = nn.GRUCell(self.input_size, self.hidden_size)
		self.h2o = nn.Linear(self.hidden_size, self.output_size)
		self.h1 = Variable(torch.zeros(batch_size, hidden_size), requires_grad=True)
		if n_layers == 2:
			self.gru2 = nn.GRUCell(self.hidden_size, self.hidden_size)
			self.h2 = Variable(torch.zeros(batch_size, hidden_size), requires_grad=True)


	def forward(self,x):
		self.h1 = self.gru1(x, self.h1)
		if self.n_layers == 2:
			self.h2 = self.gru2(self.h1, self.h2)
			o = self.h2o(self.h2)
		else:
			o = self.h2o(self.h1)

		return o

	def reset(self):
		self.h1 = Variable(torch.zeros(self.batch_size, self.hidden_size), requires_grad=True)
		if self.n_layers == 2:
			self.h2 = Variable(torch.zeros(self.batch_size, self.hidden_size), requires_grad=True)

