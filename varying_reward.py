import sys
import random
import numpy as np

def create_obstacles(width, height):
	return [(3,5),(7,5),(11,5),(3,10),(7,10),(11,10)]

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


def goal_1_reward_func(t):
	return 10

def goal_2_reward_func(t):
	return 10



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
		self.state = np.zeros(2*(self.n_obs+1),)
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
	def __init__(self, penalty, goal_1_coordinates, goal_1_func, goal_2_coordinates, goal_2_func):
		# penalty - number (integer), goal_1_coordinates - tuple, goal_1_func - lambda func returning number, goal_2_coordinates - tuple, goal_2_func - lambda function returning number
		self.terminal = False
		self.penalty = penalty
		self.goal_1_func = goal_1_func
		self.goal_2_func = goal_2_func
		self.goal_1_coordinates = goal_1_coordinates
		self.goal_2_coordinates = goal_2_coordinates
		self.t = 0 # timer

	def __call__(self, state, action, state_prime):
		self.t += 1
		if state_prime.coordinates != self.goal_1_coordinates and state_prime.coordinates != self.goal_2_coordinates:
			return self.penalty

		if state_prime.coordinates == self.goal_1_coordinates:
			self.terminal = True
			return self.goal_1_func(self.t)

		if state_prime.coordinates == self.goal_2_coordinates:
			self.terminal = True
			return self.goal_2_func(self.t)

	def reset(self, goal_1_func=None, goal_2_func=None):
		self.terminal = False
		self.t = 0
		if goal_1_func != None:
			self.goal_1_func = goal_1_func
		if goal_2_func != None:
			self.goal_2_func = goal_2_func
		
	
class TransitionFunction():
	def __init__(self, width, height, obs_func):
		# height - number (integer), width - number (integer), list_of_obstacles - list of tuples
		assert(height >= 16)
		assert(width >= 16)
		self.height = height
		self.width = width
		self.obs_func = obs_func

	def __call__(self, state, action, t):
		delta = Action.oned_to_twod(action.delta)
		t = t+1 # reward is computed later ... t+1 is the correct time to compute new obstacles
		new_list_of_obstacles = []
		obs_delta = self.obs_func(t)
		for obs in state.list_of_obstacles:
			new_obs = (obs[0] + obs_delta[0], obs[1]+obs_delta[1])
			if new_obs[0] >= self.width or new_obs[0] < 0 or new_obs[1] >= self.height or new_obs[1] < 0:
				print 'Obstacle moved outside of the grid!!!'
				sys.exit()
			new_list_of_obstacles.append(new_obs)

		# compute new coordinates here. Stay within boundary and don't move over obstacles (new).
		new_coordinates = (max(min(state.coordinates[0] + delta[0],self.width-1),0), max(min(state.coordinates[1] + delta[1],self.height-1),0))
		if new_coordinates in new_list_of_obstacles:
			# do stuff here - option 1. Remain where you are. This should be sufficient. If not, then try moving right, left down or up
			if state.coordinates not in new_list_of_obstacles:
				new_coordinates = state.coordinates # best case scenario ... stay where you are
			else:
				if (max(min(state.coordinates[0]+1,self.width-1),0), state.coordinates[1]) not in new_list_of_obstacles: # right
					new_coordinates = (max(min(state.coordinates[0]+1,self.width-1),0), state.coordinates[1])
					print 'Warning at transition 1'
				elif (max(min(state.coordinates[0]-1,self.width-1),0), state.coordinates[1]) not in new_list_of_obstacles: # left
					new_coordinates = (max(min(state.coordinates[0]-1,self.width-1),0), state.coordinates[1])
					print 'Warning at transition 2'
				elif (state.coordinates[0], max(min(state.coordinates[1]-1,self.height-1),0)) not in new_list_of_obstacles: # down
					new_coordinates = (state.coordinates[0], max(min(state.coordinates[1]-1,self.height-1),0))
					print 'Warning at transition 3'
				elif (state.coordinates[0], max(min(state.coordinates[1]+1,self.height-1),0)) not in new_list_of_obstacles: # up
					print 'Warning at transition 4'
					new_coordinates = (state.coordinates[0], max(min(state.coordinates[1]+1,self.height-1),0))
				else:
					print 'There is an obstacle for every transition!!!'
					sys.exit()

		new_state = State(new_coordinates, new_list_of_obstacles)
		return new_state


def main():
	height = 16
	width = 16

	obstacles = create_obstacles(width,height)
	s = State((0,0),obstacles)
	T = TransitionFunction(width,height,obstacle_movement)
	R = RewardFunction(penalty=-1,goal_1_coordinates=(15,0),goal_1_func=goal_1_reward_func,goal_2_coordinates=(15,15),goal_2_func=goal_2_reward_func)
	total_reward = 0
	for i in range(10000):
		a = Action(random.randint(0,4))
		t = R.t
		s_prime = T(s,a,t)
		reward = R(s,a,s_prime)
		total_reward += reward
		if R.terminal == True:
			print 'Reached goal state!'
			break
		#print 'Time: ', t
		#print 'S: ', s.coordinates
		#print 'S\': ', s_prime.coordinates
		#print 'O old: ', s.list_of_obstacles
		#print 'O new: ', s_prime.list_of_obstacles
		#print 'A: ', Action.oned_to_twod(a.delta)
		s = s_prime

	print 'Episode lasted for %d episodes.' % (i+1)
	print 'Total reward collected: ', total_reward

if __name__ == '__main__':
	main()
