import math
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.init as init


class GRU(nn.Module):
	def __init__(self, input_size, output_size, hidden_size, dtype=torch.FloatTensor, n_layers=1, batch_size=1, scale=1.0, tanh_flag=0):
		super(GRU, self).__init__()

		self.input_size = input_size
		self.output_size = output_size
		self.hidden_size = hidden_size
		self.n_layers = n_layers
		self.batch_size = batch_size
		self.scale = scale
		self.tanh_flag = tanh_flag
		self.dtype = dtype

		self.gru1 = nn.GRUCell(self.input_size, self.hidden_size)
		self.h2o = nn.Linear(self.hidden_size, self.output_size)
		init.uniform(self.h2o.weight,-3e-3, 3e-3)
		init.uniform(self.h2o.bias,-3e-3, 3e-3)
		self.h1 = Variable(torch.zeros(batch_size, hidden_size), requires_grad=True).type(dtype)
		if n_layers == 2:
			self.gru2 = nn.GRUCell(self.hidden_size, self.hidden_size)
			self.h2 = Variable(torch.zeros(batch_size, hidden_size), requires_grad=True).type(dtype)


	def forward(self,x):
		self.h1 = self.gru1(x, self.h1)
		if self.n_layers == 2:
			self.h2 = self.gru2(self.h1, self.h2)
			if self.tanh_flag:
				o = F.tanh(self.h2o(self.h2))
			else:
				o = self.h2o(self.h2)
		else:
			if self.tanh_flag:
				o = F.tanh(self.h2o(self.h1))
			else:
				o = self.h2o(self.h1)

		return self.scale*o

	def reset(self):
		self.h1 = Variable(torch.zeros(self.batch_size, self.hidden_size), requires_grad=True).type(self.dtype)
		if self.n_layers == 2:
			self.h2 = Variable(torch.zeros(self.batch_size, self.hidden_size), requires_grad=True).type(self.dtype)

