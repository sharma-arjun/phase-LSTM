import sys
import matplotlib.pyplot as plt
import numpy as np

print 'Plotting files', sys.argv[1:]

#plt.rcParams['figure.figsize'] = (10.0, 8.0)
#plt.figure(figsize=(8,6))

count = 0
for filename in sys.argv[1:]:
	f = open(filename, 'r')
	count += 1

	list_of_rewards = []
	list_of_ep_len = []

	line_count = 0
	for line in f:
		l = line.strip().split()
		line_count += 1
		if line_count > 50000:
			break
		if len(l) == 2:
			list_of_rewards.append(float(l[0]))
			list_of_ep_len.append(float(l[1]))
		else:
			break



	list_of_avg_rewards = []
	list_of_avg_ep_len = []

	for i in range(1,int(len(list_of_rewards)/500)):
		list_of_avg_rewards.append(sum(list_of_rewards[(i-1)*500:i*500])/500)

	for i in range(1,int(len(list_of_ep_len)/500)):
		list_of_avg_ep_len.append(sum(list_of_ep_len[(i-1)*500:i*500])/500)

	if count == 1:
		label = "$DRQN_{NP}$"
	elif count == 2:
		label = "$DRQN_{IP}$"
	elif count == 3:
		label = "$Phase DRQN$"
	else:
		label = count
	plt.plot(range(1, 500*len(list_of_avg_rewards)+1, 500), list_of_avg_rewards, label=label)
	#plt.scatter(range(1, len(list_of_avg_rewards)+1), list_of_avg_rewards)
	#plt.show()


	#plt.plot(range(1, len(list_of_avg_ep_len)+1), list_of_avg_ep_len)
	#plt.scatter(range(1, len(list_of_avg_ep_len)+1), list_of_avg_ep_len)
	#plt.show()

#mng = plt.get_current_fig_manager()
#mng.full_screen_toggle()

plt.xlabel('Iterations', fontsize=26)
plt.ylabel('Return', fontsize=26)
plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=3, fontsize=20)
plt.savefig('result.png',bbox_inches='tight')
plt.show()
