import math
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


def kn(p, n):
	return int((math.floor((4*p)/(2*math.pi)) + n - 1) % 4)

def spline_w(p):
	return ((4*p)/(2*math.pi)) % 1
	

class Alpha(object):
	def __init__(self):
		self._parameters = {}
		self._parameters['weight_ih'] = None
		self._parameters['weight_hh'] = None
		self._parameters['bias_ih'] = None
		self._parameters['bias_hh'] = None
		self._parameters['weight'] = None
		self._parameters['bias'] = None

		self._grad = {}
		self._grad['weight_ih'] = None
		self._grad['weight_hh'] = None
		self._grad['bias_ih'] = None
		self._grad['bias_hh'] = None
		self._grad['weight'] = None
		self._grad['bias'] = None

class PLSTM(nn.Module):
	def __init__(self, input_size, output_size, hidden_size, n_layers=1, batch_size=1):
		super(PLSTM, self).__init__()

		self.input_size = input_size
		self.output_size = output_size
		self.hidden_size = hidden_size
		self.n_layers = n_layers
		self.batch_size = batch_size

		self.control_gru_list = []
		self.control_h2o_list = []

	
		self.gru_0 = nn.GRUCell(self.input_size, self.hidden_size)
		self.h2o_0 = nn.Linear(self.hidden_size, self.output_size)
		self.gru_1 = nn.GRUCell(self.input_size, self.hidden_size)
		self.h2o_1 = nn.Linear(self.hidden_size, self.output_size)
		self.gru_2 = nn.GRUCell(self.input_size, self.hidden_size)
		self.h2o_2 = nn.Linear(self.hidden_size, self.output_size)
		self.gru_3 = nn.GRUCell(self.input_size, self.hidden_size)
		self.h2o_3 = nn.Linear(self.hidden_size, self.output_size)

		self.control_gru_list = [self.gru_0, self.gru_1, self.gru_2, self.gru_3]
		self.control_h2o_list = [self.h2o_0, self.h2o_1, self.h2o_2, self.h2o_3]

		self.alpha = []
		for i in range(4):
			self.alpha.append(Alpha())

		self.init_controls(self.control_gru_list, self.control_h2o_list, self.alpha)
		#self.gru_1 = nn.GRUCell(self.input_size, self.hidden_size)
		#self.h2o_1 = nn.Linear(self.hidden_size, self.output_size)
		self.h = Variable(torch.zeros(batch_size, hidden_size), requires_grad=True)

		self.gru_list = []
		self.h2o_list = []
		self.phase_list = []

		# to initialize grad of control gru and h2o ... I need to do this stupid thing ..
		dummy_x = Variable(torch.zeros(batch_size, input_size), requires_grad=False)
		dummy_y = Variable(torch.zeros(batch_size, output_size), requires_grad=False)
		dummy_criterion = nn.MSELoss()
		for gru, h2o in zip(self.control_gru_list, self.control_h2o_list):
			dummy_h = gru(dummy_x, self.h)
			dummy_o = h2o(dummy_h)
			dummy_loss = dummy_criterion(dummy_o, dummy_y)
			dummy_loss.backward()

		self.h = Variable(torch.zeros(batch_size, hidden_size), requires_grad=True) # reset to zero after dummy pass

	def forward(self,x,phase):
		w = self.weight_from_phase(phase, self.alpha)
		gru = nn.GRUCell(self.input_size, self.hidden_size)
		h2o = nn.Linear(self.hidden_size, self.output_size)
		self.set_weight(w, gru, h2o)
		self.gru_list.append(gru)
		self.h2o_list.append(h2o)
		self.phase_list.append(phase)

		self.h = gru(x, self.h)
		o = h2o(self.h)

		return o

	def reset(self):
		self.h = Variable(torch.zeros(self.batch_size, self.hidden_size), requires_grad=True)
		
		self.gru_list = []
		self.h2o_list = []
		self.phase_list = []

		self.init_controls(self.control_gru_list, self.control_h2o_list, self.alpha)


	def weight_from_phase(self, phase, alpha):
		weight = {}
		w = spline_w(phase)
		for key in alpha[0]._parameters.keys():
			weight[key] = alpha[kn(phase, 1)]._parameters[key] + w*0.5*(alpha[kn(phase, 2)]._parameters[key] - alpha[kn(phase, 0)]._parameters[key]) + w*w*(alpha[kn(phase, 0)]._parameters[key] - 2.5*alpha[kn(phase, 1)]._parameters[key] + 2*alpha[kn(phase, 2)]._parameters[key] - 0.5*alpha[kn(phase, 3)]._parameters[key]) + w*w*w*(1.5*alpha[kn(phase, 1)]._parameters[key] - 1.5*alpha[kn(phase, 2)]._parameters[key] + 0.5*alpha[kn(phase, 3)]._parameters[key] - 0.5*alpha[kn(phase, 0)]._parameters[key])

		return weight


	def set_weight(self, w, gru, h2o):
		gru._parameters['weight_ih'].data = w['weight_ih']
		gru._parameters['weight_hh'].data = w['weight_hh']
		gru._parameters['bias_ih'].data = w['bias_ih']
		gru._parameters['bias_hh'].data = w['bias_hh']

		h2o._parameters['weight'].data = w['weight']
		h2o._parameters['bias'].data = w['bias']

	def init_controls(self, list_of_gru, list_of_h2o, alpha):
		for i in range(len(alpha)):
			#gru = nn.GRUCell(self.input_size, self.hidden_size)
			#h2o = nn.Linear(self.hidden_size, self.output_size)
			gru = list_of_gru[i]
			h2o = list_of_h2o[i]
			#alpha[i] = {}
			alpha[i]._parameters['weight_ih'] = torch.from_numpy(np.copy(gru._parameters['weight_ih'].data.numpy()))
			alpha[i]._parameters['weight_hh'] = torch.from_numpy(np.copy(gru._parameters['weight_hh'].data.numpy()))
			alpha[i]._parameters['bias_ih'] = torch.from_numpy(np.copy(gru._parameters['bias_ih'].data.numpy()))
			alpha[i]._parameters['bias_hh'] = torch.from_numpy(np.copy(gru._parameters['bias_hh'].data.numpy()))
			alpha[i]._parameters['weight'] = torch.from_numpy(np.copy(h2o._parameters['weight'].data.numpy()))
			alpha[i]._parameters['bias'] = torch.from_numpy(np.copy(h2o._parameters['bias'].data.numpy()))

			#initialize alpha grads as zero here using shape ...
			alpha[i]._grad['weight_ih'] = torch.zeros(gru._parameters['weight_ih'].data.numpy().shape)
			alpha[i]._grad['weight_hh'] = torch.zeros(gru._parameters['weight_hh'].data.numpy().shape)
			alpha[i]._grad['bias_ih'] = torch.zeros(gru._parameters['bias_ih'].data.numpy().shape)
			alpha[i]._grad['bias_hh'] = torch.zeros(gru._parameters['bias_hh'].data.numpy().shape)
			alpha[i]._grad['weight'] = torch.zeros(h2o._parameters['weight'].data.numpy().shape)
			alpha[i]._grad['bias'] = torch.zeros(h2o._parameters['bias'].data.numpy().shape)


	def update_control_gradients(self):
		for gru, phase in zip(self.gru_list, self.phase_list):
			w = spline_w(phase)
			for key in gru._parameters.keys():
				self.alpha[kn(phase,0)]._grad[key] += gru._parameters[key].grad.data * (-0.5*w + w*w - 0.5*w*w*w)
				self.alpha[kn(phase,1)]._grad[key] += gru._parameters[key].grad.data * (1 - 2.5*w*w + 1.5*w*w*w)
				self.alpha[kn(phase,2)]._grad[key] += gru._parameters[key].grad.data * (0.5*w + 2*w*w - 1.5*w*w*w)
				self.alpha[kn(phase,3)]._grad[key] += gru._parameters[key].grad.data * (-0.5*w*w + 0.5*w*w*w)

		for h2o, phase in zip(self.h2o_list, self.phase_list):
			w = spline_w(phase)
			for key in h2o._parameters.keys():
				self.alpha[kn(phase,0)]._grad[key] += h2o._parameters[key].grad.data * (-0.5*w + w*w - 0.5*w*w*w)
				self.alpha[kn(phase,1)]._grad[key] += h2o._parameters[key].grad.data * (1 - 2.5*w*w + 1.5*w*w*w)
				self.alpha[kn(phase,2)]._grad[key] += h2o._parameters[key].grad.data * (0.5*w + 2*w*w - 1.5*w*w*w)
				self.alpha[kn(phase,3)]._grad[key] += h2o._parameters[key].grad.data * (-0.5*w*w + 0.5*w*w*w)


		for alpha, gru in zip(self.alpha, self.control_gru_list):
			for key in gru._parameters.keys():
				gru._parameters[key].grad.data = alpha._grad[key]
		for alpha, h2o in zip(self.alpha, self.control_h2o_list):
			for key in h2o._parameters.keys():
				h2o._parameters[key].grad.data = alpha._grad[key]
