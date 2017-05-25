import sys
import matplotlib.pyplot as plt
import numpy as np

filename = sys.argv[1]
f = open(filename, 'r')


list_of_rewards = []
list_of_ep_len = []

for line in f:
	l = line.strip().split()
	if len(l) == 2:
		list_of_rewards.append(int(l[0]))
		list_of_ep_len.append(int(l[1]))
	else:
		break



list_of_avg_rewards = []
list_of_avg_ep_len = []

for i in range(1,int(len(list_of_rewards)/500)):
	list_of_avg_rewards.append(sum(list_of_rewards[(i-1)*500:i*500])/500)

for i in range(1,int(len(list_of_ep_len)/500)):
	list_of_avg_ep_len.append(sum(list_of_ep_len[(i-1)*500:i*500])/500)


plt.plot(range(1, len(list_of_avg_rewards)+1), list_of_avg_rewards)
plt.scatter(range(1, len(list_of_avg_rewards)+1), list_of_avg_rewards)
plt.show()


plt.plot(range(1, len(list_of_avg_ep_len)+1), list_of_avg_ep_len)
plt.scatter(range(1, len(list_of_avg_ep_len)+1), list_of_avg_ep_len)
plt.show()
