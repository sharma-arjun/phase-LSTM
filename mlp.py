import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class MLP(nn.Module):
	def __init__(self, input_size, output_size, hidden_size, n_layers=1):
		super(MLP, self).__init__()

		self.input_size = input_size
		self.output_size = output_size
		self.hidden_size = hidden_size
		self.n_layers = n_layers

		self.l1 = nn.Linear(self.input_size, self.hidden_size)
		if n_layers == 2:
			self.l2 = nn.Linear(self.hidden_size, self.hidden_size)
		self.h2o = nn.Linear(self.hidden_size, self.output_size)


	def forward(self,x):
		h0 = F.relu(self.l1(x))
		if self.n_layers == 2:
			h1 = F.relu(self.l2(h0))
			o = self.h2o(h1)
		else:
			o = self.h2o(h0)

		return o

	def reset(self):
		pass
