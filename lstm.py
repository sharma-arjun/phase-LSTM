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


		self.gru = nn.GRUCell(self.input_size, self.hidden_size)
		self.h2o = nn.Linear(self.hidden_size, self.output_size)
		self.h = Variable(torch.zeros(batch_size, hidden_size), requires_grad=True)


	def forward(self,x):
		self.h = self.gru(x, self.h)
		o = self.h2o(self.h)

		return o

	def reset(self):
		self.h = Variable(torch.zeros(self.batch_size, self.hidden_size), requires_grad=True)

