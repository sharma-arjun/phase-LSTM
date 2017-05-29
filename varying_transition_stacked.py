import sys
import copy
import math
import random
import numpy as np
from phase_lstm_multilayer import PLSTM
from lstm import LSTM
from mlp import MLP
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from itertools import product

def create_obstacles(width, height):
	#return [(4,6),(9,6),(14,6),(4,12),(9,12),(14,12)] # 19 x 19
	#return [(3,5),(7,5),(11,5),(3,10),(7,10),(11,10)] # 17 x 17
	#return [(3,4),(6,4),(9,4),(3,9),(6,9),(9,9)] # 15 x 15
	#return [(4,4),(7,4),(4,8),(7,8)] # 13 x 13
	return [(3,3),(6,3),(3,6),(6,6)] # 12 x 12
	#return [] # no obstacles

def obstacle_movement(t):
	if t % 6 == 0:
		return (0,1) # move up
	elif t % 6 == 1:
		return (1,0) # move right
	elif t % 6 == 2:
		return (1,0) # move right
	elif t % 6 == 3:
		return (0,-1) # move down
	elif t % 6 == 4:
		return (-1,0) # move left
	elif t % 6 == 5:
		return (-1, 0) # move left


def create_targets(memory, q_vals, target_net, policy_type, gamma=1):
	# memory: 0 - set of current_states 1: action index 2: reward 3: next state 4: phase 5: phase_prime
	n_eps = len(memory)
	action_space_size = target_net.output_size 
	q_target = torch.zeros((n_eps, action_space_size))
	
	for i in range(n_eps):
		phase_prime = memory[i][5]
		s_1 = memory[i][0][1]
		s = memory[i][0][2]
		s_prime = memory[i][3]
		if policy_type == 0:
			inp = np.concatenate((s_1.state, s.state, s_prime.state))
			x = Variable(torch.from_numpy(inp).float(), requires_grad=False).unsqueeze(0)
			q_prime = target_net.forward(x)
		elif policy_type == 1:
			inp = np.concatenate((s_1.state, s.state, s_prime.state, np.asarray([phase_prime])))
			x = Variable(torch.from_numpy(inp).float(), requires_grad=False).unsqueeze(0)
			q_prime = target_net.forward(x)
		elif policy_type == 2:
			inp = np.concatenate((s_1.state, s.state, s_prime.state))
			x = Variable(torch.from_numpy(inp).float(), requires_grad=False).unsqueeze(0)
			q_prime = target_net.forward(x, phase_prime)
		elif policy_type == 3:
			inp = np.concatenate((s_1.state, s.state, s_prime.state))
			x = Variable(torch.from_numpy(inp).float(), requires_grad=False).unsqueeze(0)
			q_prime = target_net.forward(x)
		elif policy_type == 4:
			inp = np.concatenate((s_1.state, s.state, s_prime.state, np.asarray([phase_prime])))
			x = Variable(torch.from_numpy(inp).float(), requires_grad=False).unsqueeze(0)
			q_prime = target_net.forward(x)

		q_target[i,:] = q_vals[i][0,:].data.clone()
		q_target[i, memory[i][1]] = gamma*(memory[i][2] + q_prime.data[0,np.argmax(q_prime.data.numpy())])

	target_net.reset()
	return q_target


def goal_1_reward_func(w,t,p):
	#return -20*math.sin(w*t + p) + 5
	return -20

def goal_2_reward_func(w,t,p):
	#return 20*math.sin(w*t + p) + 5
	return 20


class State():
	def __init__(self, coordinates, list_of_obstacles):
		#coordinates - tuple, list_of_obstacles - list of tuples
		assert(len(coordinates) == 2)
		self.coordinates = coordinates
		self.n_obs = 0
		for obs in list_of_obstacles:
			assert(len(obs) == 2)
			self.n_obs += 1
		
		self.list_of_obstacles = list_of_obstacles
		self.state = np.zeros(2*(self.n_obs+1))
		self.state[0] = self.coordinates[0]
		self.state[1] = self.coordinates[1]
		for i in range(1,len(list_of_obstacles)+1):
			self.state[2*i] = list_of_obstacles[i-1][0]
			self.state[2*i+1] = list_of_obstacles[i-1][1]
		

class Action():
	def __init__(self, delta):
		#delta - number (integer)
		assert(delta in (0,1,2,3,4))
		self.delta = delta

	@staticmethod
	def oned_to_twod(delta):
		assert(delta in (0,1,2,3,4))
		if delta == 0:
			return (0,0) # no movement
		elif delta == 1:
			return (0,1) # up
		elif delta == 2:
			return (0,-1) # down
		elif delta == 3:
			return (-1,0) # left
		elif delta == 4:
			return (1,0) # right



class RewardFunction():
	def __init__(self, penalty, goal_1_coordinates, goal_1_func, goal_2_coordinates, goal_2_func, w1=None, w2=None):
		# penalty - number (integer), goal_1_coordinates - tuple, goal_1_func - lambda func returning number, goal_2_coordinates - tuple, goal_2_func - lambda function returning number
		self.terminal = False
		self.penalty = penalty
		self.goal_1_func = goal_1_func
		self.goal_2_func = goal_2_func
		self.goal_1_coordinates = goal_1_coordinates
		self.goal_2_coordinates = goal_2_coordinates
		self.t = 0 # timer
		self.w1 = w1
		self.w2 = w2
		self.p = None
		

	def __call__(self, state, action, state_prime):
		self.t += 1
		if state_prime.coordinates != self.goal_1_coordinates and state_prime.coordinates != self.goal_2_coordinates:
			return self.penalty

		if state_prime.coordinates == self.goal_1_coordinates:
			self.terminal = True
			return self.goal_1_func(self.w1, self.t, self.p)

		if state_prime.coordinates == self.goal_2_coordinates:
			self.terminal = True
			return self.goal_2_func(self.w2, self.t, self.p)

	def reset(self, goal_1_func=None, goal_2_func=None):
		self.terminal = False
		self.t = 0

		if goal_1_func != None:
			self.goal_1_func = goal_1_func
		if goal_2_func != None:
			self.goal_2_func = goal_2_func


class TransitionFunction():
	def __init__(self, width, height, obs_func, w, prob=0.1):
		# height - number (integer), width - number (integer), list_of_obstacles - list of tuples
		#assert(height >= 16)
		#assert(width >= 16)
		self.height = height
		self.width = width
		self.obs_func = obs_func
		self.w = w # controls how often phase changes ... phase will change after every w time steps
		self.p = 0 # later select randomly between 0, pi/2, pi, 3pi/2, 2pi
		self.prob = prob # probability with which agent moves with the wind

	def __call__(self, state, action,t):
		delta = Action.oned_to_twod(action.delta)
		t = t + 1 # one more than reward because reward is called after transition and t is maintained by reward. t maintained by reward for easy reset.
		new_list_of_obstacles = []
		obs_delta = self.obs_func(t)
		for obs in state.list_of_obstacles:
			new_obs = (obs[0] + obs_delta[0], obs[1]+obs_delta[1])
			if new_obs[0] >= self.width or new_obs[0] < 0 or new_obs[1] >= self.height or new_obs[1] < 0:
				print 'Obstacle moved outside of the grid!!!'
				sys.exit()
			new_list_of_obstacles.append(new_obs)

		# internal phase
		phase = self.phase(t-1) # phase computed on current time t not t+1
		#change delta based on internal phase
		if phase == 0: #up
			if np.random.uniform() < self.prob:
				delta = (0,1)
		elif phase == math.pi/2: # right
			if np.random.uniform() < self.prob:
				delta = (1,0)
		elif phase == math.pi: # down
			if np.random.uniform() < self.prob:
				delta = (0,-1)
		elif phase == 3*math.pi/2: # left
			if np.random.uniform() < self.prob:
				delta = (-1,0)
		else:
			print 'Unknown phase'
			sys.exit()
		# compute new coordinates here. Stay within boundary and don't move over obstacles (new).
		new_coordinates = (max(min(state.coordinates[0] + delta[0],self.width-1),0), max(min(state.coordinates[1] + delta[1],self.height-1),0))
		if new_coordinates in new_list_of_obstacles:
			# do stuff here - option 1. Remain where you are. This should be sufficient. If not, then try moving right, left down or up
			if state.coordinates not in new_list_of_obstacles:
				new_coordinates = state.coordinates # best case scenario ... stay where you are
			else:
				if (max(min(state.coordinates[0]+1,self.width-1),0), state.coordinates[1]) not in new_list_of_obstacles: # right
					new_coordinates = (max(min(state.coordinates[0]+1,self.width-1),0), state.coordinates[1])
					#print 'Warning at transition 1'
				elif (max(min(state.coordinates[0]-1,self.width-1),0), state.coordinates[1]) not in new_list_of_obstacles: # left
					new_coordinates = (max(min(state.coordinates[0]-1,self.width-1),0), state.coordinates[1])
					#print 'Warning at transition 2'
				elif (state.coordinates[0], max(min(state.coordinates[1]-1,self.height-1),0)) not in new_list_of_obstacles: # down
					new_coordinates = (state.coordinates[0], max(min(state.coordinates[1]-1,self.height-1),0))
					#print 'Warning at transition 3'
				elif (state.coordinates[0], max(min(state.coordinates[1]+1,self.height-1),0)) not in new_list_of_obstacles: # up
					#print 'Warning at transition 4'
					new_coordinates = (state.coordinates[0], max(min(state.coordinates[1]+1,self.height-1),0))
				else:
					print 'There is an obstacle for every transition!!!'
					sys.exit()

		new_state = State(new_coordinates, new_list_of_obstacles)
		return new_state

	def phase(self,t):
		return ((math.floor(t/self.w)/2 + (self.p/math.pi)) % 2)*math.pi


class ExperienceReplay():
	def __init__(self, max_memory_size = 100):
		self.memory = []
		self.oldest = -1
		self.max_memory_size = max_memory_size
	
	def add(self, experience):
		if len(self.memory) < self.max_memory_size: 
			self.memory.append(experience)
			self.oldest = 0
		else:
			self.memory.insert(self.oldest, experience)
			self.oldest = (self.oldest + 1) % self.max_memory_size

	def sample(self):
		idx = np.random.randint(0, high=len(self.memory))
		return self.memory[idx]
			


def epsilon_greedy_linear_decay(action_vector, n_episodes, n, low=0.1, high=0.9):
	if n <= n_episodes:
		eps = ((low-high)/n_episodes)*n + high
	else:
		eps = low

	if np.random.uniform() > eps:
		return np.argmax(action_vector)
	else:
		return np.random.randint(low=0, high=5)

def epsilon_greedy(action_vector, eps):
	if np.random.uniform() > eps:
		return np.argmax(action_vector)
	else:
		return np.random.randint(low=0, high=5)

def sample_start(set_diff):
	return random.choice(set_diff)

def main():
	height = width = 12
	max_episode_length = 600
	n_episodes = 50000
	n_copy_after = 1000
	burn_in = 100
	policy_type = int(sys.argv[1])

	obstacles = create_obstacles(width,height)

	set_diff = list(set(product(tuple(range(width)), repeat=2)) - set(obstacles))
	#start_loc = (0,5)
	start_loc = sample_start(set_diff)
	s = State(start_loc,obstacles)
	T = TransitionFunction(width,height,obstacle_movement,8)
	R = RewardFunction(penalty=-1,goal_1_coordinates=(11,0),goal_1_func=goal_1_reward_func,goal_2_coordinates=(11,11),goal_2_func=goal_2_reward_func, w1=math.pi/8, w2=math.pi/8)
	M = ExperienceReplay(max_memory_size=1000)
	
	if policy_type == 0: # rnn without phase
		policy = LSTM(input_size=s.state.shape[0]*3, output_size=5, hidden_size=8, n_layers=2, batch_size=1)
	elif policy_type == 1: # rnn with phase as additional input
		policy = LSTM(input_size=s.state.shape[0]*3+1, output_size=5, hidden_size=8, n_layers=2, batch_size=1)
	elif policy_type == 2: # phase rnn
		policy = PLSTM(input_size=s.state.shape[0]*3, output_size=5, hidden_size=8, n_layers=2, batch_size=1)
	elif policy_type == 3: # mlp without phase
		policy = MLP(input_size=s.state.shape[0]*3, output_size=5, hidden_size=8, n_layers=2, batch_size=1)
	elif policy_type == 4: # mlp with phase as additional input
		policy = MLP(input_size=s.state.shape[0]*3+1, output_size=5, hidden_size=8, n_layers=2, batch_size=1)

	target_net = copy.deepcopy(policy)
	criterion = nn.MSELoss()
	optimizer = optim.Adam(policy.parameters(), lr=0.0001)

	list_of_total_rewards = []
	list_of_n_episodes = []

	s_2 = State(start_loc,obstacles)
	s_1 = State(start_loc,obstacles)
	#Burn in with random policy
	for i in range(burn_in):
		episode_experience = []
		for j in range(max_episode_length):
			#x = Variable(torch.from_numpy(s.state).float(), requires_grad=False).unsqueeze(0)
			#q = policy.forward(x)
			a = Action(np.random.randint(0,high=5))
			#a = Action(epsilon_greedy_linear_decay(q.data.numpy(),10000, i))
			#a = Action(epsilon_greedy(q.data.numpy(), 0.1))
			t = R.t
			phase = T.phase(t)
			phase_prime = T.phase(t+1)
			s_prime = T(s,a,t)
			reward = R(s,a,s_prime)
			episode_experience.append(((s_2, s_1, s), a.delta, reward, s_prime, phase, phase_prime))
			if R.terminal == True:
				#print 'Reached goal state!'
				break
			s_2 = copy.deepcopy(s_1)
			s_1 = copy.deepcopy(s)
			s = s_prime

		M.add(episode_experience)
		R.reset()
		start_loc = sample_start(set_diff)
		s = State(start_loc,obstacles)

	print 'Burn in completed'

	filename = 'plotfiles/' + sys.argv[2]
	print 'Writing to ' + filename
	f = open(filename,'w')

	for i in range(n_episodes):
		total_reward = 0
		episode_experience = []
		# zero gradients
		optimizer.zero_grad()
		for j in range(max_episode_length):
			phase = T.phase(R.t)
			if policy_type == 0:
				inp = np.concatenate((s_2.state, s_1.state, s.state))
				x = Variable(torch.from_numpy(inp).float(), requires_grad=False).unsqueeze(0)
				q = policy.forward(x)
			elif policy_type == 1:
				inp = np.concatenate((s_2.state, s_1.state, s.state, np.asarray([phase])))
				x = Variable(torch.from_numpy(inp).float(), requires_grad=False).unsqueeze(0)
				q = policy.forward(x)
			elif policy_type == 2:
				inp = np.concatenate((s_2.state, s_1.state, s.state))
				x = Variable(torch.from_numpy(inp).float(), requires_grad=False).unsqueeze(0)
				q = policy.forward(x, phase)
			elif policy_type == 3:
				inp = np.concatenate((s_2.state, s_1.state, s.state))
				x = Variable(torch.from_numpy(inp).float(), requires_grad=False).unsqueeze(0)
				q = policy.forward(x)
			elif policy_type == 4:
				inp = np.concatenate((s_2.state, s_1.state, s.state, np.asarray([phase])))
				x = Variable(torch.from_numpy(inp).float(), requires_grad=False).unsqueeze(0)
				q = policy.forward(x)

			a = Action(epsilon_greedy_linear_decay(q.data.numpy(), 25000, i))
			#a = Action(epsilon_greedy(q.data.numpy(), 0.1))
			t = R.t
			s_prime = T(s,a,t)
			reward = R(s,a,s_prime)
			total_reward += reward
			phase_prime = T.phase(R.t)
			episode_experience.append(((s_2, s_1, s), a.delta, reward, s_prime, phase, phase_prime))
			if R.terminal == True:
				#print 'Reached goal state!'
				break
			#q_vals.append(q)
			s_2 = copy.deepcopy(s_1)
			s_1 = copy.deepcopy(s)
			s = s_prime

		M.add(episode_experience)
		#print 'Episode lasted for %d steps.' % (j+1)
		#print 'Total reward collected: ', total_reward
		list_of_total_rewards.append(total_reward)
		list_of_n_episodes.append(j+1)
		if i % 500 == 0 and i > 0:
			print str(i) + ': Avg. Reward: ' + str(sum(list_of_total_rewards[i-500:i])/500.0) + ' Avg. Episode length: ' + str(sum(list_of_n_episodes[i-500:i])/500.0)

		# write to file for plotting
		f.write(str(total_reward) + ' ' + str(j+1) + '\n')

		policy.reset()

		# save policy
		if i % 1000 == 0 and i> 0:
			checkpoint_name = 'checkpoints/' + sys.argv[3] + '_' + str(i) + '.pth'
			f_w = open(checkpoint_name, 'wb')
			torch.save(policy,f_w)

		# forward pass through memory sample
		memory = M.sample()
		q_vals = []
		for j in range(len(memory)):
			s_2 = memory[j][0][0]
			s_1 = memory[j][0][1]
			s = memory[j][0][2]
			phase = memory[j][4]
			if policy_type == 0:
				inp = np.concatenate((s_2.state, s_1.state, s.state))
				x = Variable(torch.from_numpy(inp).float(), requires_grad=False).unsqueeze(0)
				q = policy.forward(x)
			elif policy_type == 1:
				inp = np.concatenate((s_2.state, s_1.state, s.state, np.asarray([phase])))
				x = Variable(torch.from_numpy(inp).float(), requires_grad=False).unsqueeze(0)
				q = policy.forward(x)
			elif policy_type == 2:
				inp = np.concatenate((s_2.state, s_1.state, s.state))
				x = Variable(torch.from_numpy(inp).float(), requires_grad=False).unsqueeze(0)
				q = policy.forward(x, phase)
			elif policy_type == 3:
				inp = np.concatenate((s_2.state, s_1.state, s.state))
				x = Variable(torch.from_numpy(inp).float(), requires_grad=False).unsqueeze(0)
				q = policy.forward(x)
			elif policy_type == 4:
				inp = np.concatenate((s_2.state, s_1.state, s.state, np.asarray([phase])))
				x = Variable(torch.from_numpy(inp).float(), requires_grad=False).unsqueeze(0)
				q = policy.forward(x)

			q_vals.append(q)

		# backward pass
		targets = Variable(create_targets(memory, q_vals, target_net, policy_type, gamma=1), requires_grad=False)
		outputs = torch.stack(q_vals,0).squeeze(1)
		loss = criterion(outputs, targets)
		loss.backward(retain_variables=False)

		# clip gradients here ...
		nn.utils.clip_grad_norm(policy.parameters(), 5.0)
		for p in policy.parameters():
			p.data.add_(0.0001, p.grad.data)

		# phase lstm step
		if policy_type == 2:
			policy.update_control_gradients()

		# optimizer step
		optimizer.step()

		# Reset environment and policy hidden vector at the end of episode
		policy.reset()
		R.reset()
		start_loc = sample_start(set_diff)
		s = State(start_loc,obstacles)

		# copy into target network
		if i % n_copy_after == 0 and i > 0:
			target_net = copy.deepcopy(policy)


	# testing with greedy policy
	print 'Using greedy policy ...'
	start_loc = (0,5)
	s_2 = State(start_loc, obstacles)
	s_1 = State(start_loc, obstacles)
	s = State(start_loc, obstacles)
	R.reset()
	total_reward = 0
	step_count = 0
	while R.terminal == False:
		phase = T.phase(R.t)
		if policy_type == 0:
			inp = np.concatenate((s_2.state, s_1.state, s.state))
			x = Variable(torch.from_numpy(inp).float(), requires_grad=False).unsqueeze(0)
			q = policy.forward(x)
		elif policy_type == 1:
			inp = np.concatenate((s_2.state, s_1.state, s.state, np.asarray([phase])))
			x = Variable(torch.from_numpy(inp).float(), requires_grad=False).unsqueeze(0)
			q = policy.forward(x)
		elif policy_type == 2:
			inp = np.concatenate((s_2.state, s_1.state, s.state))	
			x = Variable(torch.from_numpy(inp).float(), requires_grad=False).unsqueeze(0)
			q = policy.forward(x, phase)
		elif policy_type == 3:
			inp = np.concatenate((s_2.state, s_1.state, s.state))
			x = Variable(torch.from_numpy(inp).float(), requires_grad=False).unsqueeze(0)
			q = policy.forward(x)
		elif policy_type == 4:
			inp = np.concatenate((s_2.state, s_1.state, s.state, np.asarray([phase])))
			x = Variable(torch.from_numpy(inp).float(), requires_grad=False).unsqueeze(0)
			q = policy.forward(x)
		a = Action(np.argmax(q.data.numpy()))
		t = R.t
		s_prime = T(s,a,t)
		reward = R(s,a,s_prime)
		total_reward += reward
		step_count += 1
		s_2 = copy.deepcopy(s_1)
		s_1 = copy.deepcopy(s)
		s = s_prime

	print 'Total reward', total_reward
	print 'Number of steps', step_count

	f.close()

if __name__ == '__main__':
	main()
