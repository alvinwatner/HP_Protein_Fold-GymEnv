from simulation import environment
import numpy as np

env = environment()
current_state = env.reset(amino_input = ['P', 'P', 'P', 'H', 'H', 'P', 'P', 'H', 'H', 'P', 'P', 'P', 'P', 'P', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'P', 'P', 'H', 'H', 'P', 'P', 'P', 'P', 'H', 'H', 'P', 'P', 'H', 'P', 'P'])
done = False

while not done:
	action = np.random.randint(0, env.action_space_size)
	new_state, reward, done = env.step(action)	
	# env.render()
env.render(plot = True) ##show result figure only