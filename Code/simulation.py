"""
	=====================================================
	Protein Folding 2D Hydrophobic-Polar Model Simulation
	=====================================================

	Protein Folding is well known optimization combinatorial problem, there are some models to adress the folding process.
	There are 20 different amino acid, Hydrophobic-Polar(HP) Model classified those amino into 2 types : H(Hydrophobic) and P(Hydrophillic).
	HP Model is one of my favorite, since it more looks like a board game with a set of simple rules, But yea the simplicity also determined as NP-complete problem.
	Here's how it works :
	1. Given a set of amino 'H' and 'P' sequence.
	2. Place all the the sequence one by one to 2D (or3D space). 
	3. Amino should be placed adjacent to the previous amino (Up, Left, Right, or Down). (note: placing to occupied occupied is not allowed).
	Goals is to find H-H pairs that not connected to primary structure but consecutive in 2D space. 

	Thats it! sounds confusing??  no worry, Lets roll over..
	
	The following code, follows the OpenAI gym based environment.

	One things you should know is the observation_value return a list : [amino_data, image_data]
	- amino_data = list with size 100 (to contain max amino size, 100) 
	- image_data = RGB image (150,150,3)  

----------------------------------------
Author  : Alvin Watner
Email   : alvin2phantomhive@gmail.com
Website : -
License : -
-----------------------------------------
**Please feel free to use and modify this, but keep the above information. Thanks!**
"""

import numpy as np
from PIL import Image
import cv2

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

class environment():
	def __init__(self):	
		# Since it was 2 dimensional environment, it consist 4 possible action given a state 		
		self.action_space = np.array([0, 1, 2, 3]) # 0 = up, 1 = left, 2 = right, 3 = down
		self.action_space_size = len(self.action_space)

		self.win_size = 6000 #Window Size (Yea, i know it was a huge image, but this size could handle 100 amino sequence, with acceptable molecul size)

		#initialize the amino coordinate position
		self.init_amino_position_x = int(self.win_size/2)
		self.init_amino_position_y = int(self.win_size/2)

		#Follows gym environment, Isreset = True if environment.reset() is called, otherwise it raise an error
		self.Isreset = False

	def adjust_amino_size(self):
		#Default amino size, it will be reduces as the amino sequence get longer
		default_size = 80

		center = self.win_size/2
		max_size = (default_size *3) * len(self.amino_acid) #default_size * 3, because the value is the radius of the circle not diameter (also make it larger a bit)

		"""
		Since the 'init_amino_position' always start at the center of the window, codes below used to check if (default_size * amino length) is go beyond the window size 
		and shrink the 'default_size' if it does.
		"""

		if max_size > center:
			while max_size > center:
				#bound the minimum size to 11 
				if default_size < 11:
					return default_size

				default_size -= 1
				max_size = (default_size * 3) * len(self.amino_acid)
			return default_size
		else:
			return default_size

	def preprocess_data(self, amino_data):
		"""
		Convert amino_data to number for ease computation
		-H (Hydrophobic)  = 1
		-P (Hydrophillic) = 2
		"""
		for i in range(len(amino_data)):
			if amino_data[i] == 'H':
				amino_data[i] = 1
			elif amino_data[i] == 'P':
				amino_data[i] = 2

		return np.array(amino_data)

	"""
	Codes Below Are The Drawing Process
	Nothing fancy, just using regular opencv functionality
	"""
		
	def draw_amino(self, amino_type = None, coordinat_x = 0, coordinat_y = 0, size = 0):
		"""
		Draw amino :
		- Hydrophobic Amino Acid  = Black Circle
		- Hydrophillic Amino Acid = White Circle
		"""
		if amino_type == 1:
			return cv2.circle(self.current_image, (coordinat_x, coordinat_y), size , (0,0,0) , -2) 
		else:
			return cv2.circle(self.current_image, (coordinat_x, coordinat_y), size , (255,255,255) , -2)

	def draw_arrow_line(self, start_point = (0, 0), end_point = (2, 2)):
		#draw a line and arrow pointing to the next amino in the sequence
		return cv2.arrowedLine(self.current_image, start_point, end_point, (255,0,0), 4) 

	def draw_next_amino(self, amino_type = None, prev_coordinat_x = 0, prev_coordinat_y = 0, size = 0, action = 0):
		"""
		This Function draws next amino acid from the sequence, when step() function being called.
		The Rule is : Next amino acid always placed consecutive to the previous amino

		Parameter :
		- amino_type       = int, 1(Hydrophobic) or 2(Hydrophillic)
		- prev_coordinat_x = int, amino coordinate x axis 
		- prev_coordinat_y = int, amino coordinate y axis
		- size             = int, amino size 
		- action           = int, action from action_space (1, 2, 3, 4)

		Return :
		- amino coordinate x axis
		- amino coordinate y axis
		- RGB image
		"""

		if amino_type == 1: 
			if action == 0:
				new_amino_position_x, new_amino_position_y, img = self.draw_next_up(amino_type = 1, coor_x = prev_coordinat_x, coor_y = prev_coordinat_y)
				return new_amino_position_x, new_amino_position_y, img
			elif action == 1:
				new_amino_position_x, new_amino_position_y, img = self.draw_next_left(amino_type = 1, coor_x = prev_coordinat_x, coor_y = prev_coordinat_y)
				return new_amino_position_x, new_amino_position_y, img
			elif action == 2:
				new_amino_position_x, new_amino_position_y, img = self.draw_next_right(amino_type = 1, coor_x = prev_coordinat_x, coor_y = prev_coordinat_y)
				return new_amino_position_x, new_amino_position_y, img
			else:
				new_amino_position_x, new_amino_position_y, img = self.draw_next_down(amino_type = 1, coor_x = prev_coordinat_x, coor_y = prev_coordinat_y)
				return new_amino_position_x, new_amino_position_y, img
		
		elif amino_type == 2 :
			if action == 0:
				new_amino_position_x, new_amino_position_y, img = self.draw_next_up(amino_type = 2, coor_x = prev_coordinat_x, coor_y = prev_coordinat_y)
				return new_amino_position_x, new_amino_position_y, img
			elif action == 1:
				new_amino_position_x, new_amino_position_y, img = self.draw_next_left(amino_type = 2, coor_x = prev_coordinat_x, coor_y = prev_coordinat_y)
				return new_amino_position_x, new_amino_position_y, img
			elif action == 2:
				new_amino_position_x, new_amino_position_y, img = self.draw_next_right(amino_type = 2, coor_x = prev_coordinat_x, coor_y = prev_coordinat_y)
				return new_amino_position_x, new_amino_position_y, img
			else:
				new_amino_position_x, new_amino_position_y, img = self.draw_next_down(amino_type = 2, coor_x = prev_coordinat_x, coor_y = prev_coordinat_y)
				return new_amino_position_x, new_amino_position_y, img

	def draw_next_up(self, amino_type = None, coor_x = 0, coor_y = 0):
		#Return New coordinate and RGB image, after action '0 : up'
		new_amino_position_x = coor_x
		new_amino_position_y = coor_y - self.amino_move
		start_line_x = coor_x
		start_line_y = coor_y - self.line_length		

		img = self.draw_arrow_line(start_point = (start_line_x, start_line_y), end_point = (start_line_x, start_line_y - self.line_length))
		img = self.draw_amino(amino_type = amino_type, coordinat_x = new_amino_position_x, coordinat_y = new_amino_position_y, size = self.amino_size)

		return new_amino_position_x, new_amino_position_y, img

	def draw_next_left(self, amino_type = None, coor_x = 0, coor_y = 0):
		#Return New coordinate and RGB image, after action '1 : left'
		new_amino_position_x = coor_x - self.amino_move	
		new_amino_position_y = coor_y
		start_line_x = coor_x - self.line_length
		start_line_y = coor_y		
		
		img = self.draw_arrow_line(start_point = (start_line_x, start_line_y), end_point = (start_line_x - self.line_length, start_line_y))
		img = self.draw_amino(amino_type = amino_type, coordinat_x = new_amino_position_x, coordinat_y = new_amino_position_y, size = self.amino_size)

		return new_amino_position_x, new_amino_position_y, img

	def draw_next_right(self, amino_type = None, coor_x = 0, coor_y = 0):
		#Return New coordinate and RGB image, after action '2 : right'
		new_amino_position_x = coor_x + self.amino_move			
		new_amino_position_y = coor_y
		start_line_x = coor_x + self.line_length
		start_line_y = coor_y		
		
		img = self.draw_arrow_line(start_point = (start_line_x, start_line_y), end_point = (start_line_x + self.line_length, coor_y))
		img = self.draw_amino(amino_type = amino_type, coordinat_x = new_amino_position_x, coordinat_y = new_amino_position_y, size = self.amino_size)

		return new_amino_position_x, new_amino_position_y, img

	def draw_next_down(self, amino_type = None, coor_x = 0, coor_y = 0):
		#Return New coordinate and RGB image, after action '3 : down'
		new_amino_position_x = coor_x
		new_amino_position_y = coor_y + self.amino_move					
		start_line_x = coor_x 
		start_line_y = coor_y + self.line_length		
		
		img = self.draw_arrow_line(start_point = (start_line_x, start_line_y), end_point = (coor_x, start_line_y + self.line_length))
		img = self.draw_amino(amino_type = amino_type, coordinat_x = new_amino_position_x, coordinat_y = new_amino_position_y, size = self.amino_size)

		return new_amino_position_x, new_amino_position_y, img
	
	"""
	Codes Below Use To Check Current Amino Neighbour
	The Functions Returns:
	----------------------
	- Free Energy = Int, '-1' if the neighbour is Hydrophobic and Not Connected in Primary Structure (arrow line), '0' otherwise
	- Amino       = Bool, 'True' if the neighbour of current amino (based on given coordinate) exist another amino, 'False' if there is no amino
	"""

	def check_Above_Neighbour(self, new_coordinat_x, new_coordinat_y):
		half_line_length = int(0.5 * self.line_length) 
		
		#Check if above neighbour exist hydrophobic amino
		if np.sum(self.current_image[new_coordinat_y - self.amino_move, new_coordinat_x]) == 0:
			amino = True
			#Then check if it is connected or not
			if np.sum(self.current_image[new_coordinat_y - half_line_length * 3, new_coordinat_x]) == 255:
				free_energy = 0
			else:
				free_energy = -1

		#Check if above neighbour exist hydrophillic amino
		elif np.sum(self.current_image[new_coordinat_y - self.amino_move, new_coordinat_x]) == 765:
			amino = True
			free_energy = 0
		#Check if above neighbour exist nothing
		elif np.sum(self.current_image[new_coordinat_y - self.amino_move, new_coordinat_x]) == 330:
			amino = False
			free_energy = 0

		return free_energy, amino		

	def check_Left_Neighbour(self, new_coordinat_x, new_coordinat_y):
		half_line_length = int(0.5 * self.line_length)

		#Check if left neighbour exist hydrophobic amino
		if np.sum(self.current_image[new_coordinat_y, new_coordinat_x - self.amino_move]) == 0:
			amino = True
			#Then check if it is connected or not
			if np.sum(self.current_image[new_coordinat_y, new_coordinat_x - half_line_length * 3]) == 255:
				free_energy = 0
			else:
				free_energy = -1
				
		#Check if left neighbour exist hydrophillic amino
		elif np.sum(self.current_image[new_coordinat_y, new_coordinat_x - self.amino_move]) == 765:
			amino = True
			free_energy = 0
		#Check if left neighbour exist nothing
		elif np.sum(self.current_image[new_coordinat_y, new_coordinat_x - self.amino_move]) == 330:
			amino = False
			free_energy = 0

		return free_energy, amino

	def check_Right_Neighbour(self, new_coordinat_x, new_coordinat_y):
		half_line_length = int(0.5 * self.line_length)

		#Check if right neighbour exist hydrophobic amino
		if np.sum(self.current_image[new_coordinat_y, new_coordinat_x + self.amino_move]) == 0:
			amino = True
			#Then check if it is connected or not
			if np.sum(self.current_image[new_coordinat_y, new_coordinat_x + half_line_length * 3]) == 255:
				free_energy = 0
			else:
				free_energy = -1
				
		#Check if right neighbour exist hydrophillic amino
		elif np.sum(self.current_image[new_coordinat_y, new_coordinat_x + self.amino_move]) == 765:
			amino = True
			free_energy = 0
		#Check if right neighbour exist nothing
		elif np.sum(self.current_image[new_coordinat_y, new_coordinat_x + self.amino_move]) == 330:
			amino = False
			free_energy = 0

		return free_energy, amino
	
	def check_Below_Neighbour(self, new_coordinat_x, new_coordinat_y):

		half_line_length = int(0.5 * self.line_length)

		#Check if below neighbour exist hydrophobic amino
		if np.sum(self.current_image[new_coordinat_y + self.amino_move, new_coordinat_x]) == 0:
			amino = True
			#Then check if it is connected or not
			if np.sum(self.current_image[new_coordinat_y + half_line_length * 3, new_coordinat_x]) == 255:
				free_energy = 0
			else:
				free_energy = -1

		#Check if below neighbour exist hydrophillic amino
		elif np.sum(self.current_image[new_coordinat_y + self.amino_move, new_coordinat_x]) == 765:
			amino = True
			free_energy = 0
		#Check if below neighbour exist nothing
		elif np.sum(self.current_image[new_coordinat_y + self.amino_move, new_coordinat_x]) == 330:
			amino = False
			free_energy = 0

		return free_energy, amino

	"""
	Codes Below Use To Generate Random Amino
		Parameter : int, num_sequence(Optional) : amino length 
		Return    : list, (eg: ['H', 'H', 'P', ... 'H or P'])
	"""

	def generate_Random_Amino(self, num_sequence = 50):
		random_amino = []
		rand = 0.5
		for i in range(num_sequence):
			if random.random() > rand:
				random_amino.append('H')
			else:
				random_amino.append('P')

		return random_amino

	"""
	Codes Below Use Calculate The Reward 
		Reward Formula  | Collision Punishment = -2 | Trap Punishment = -4 |
		==============
		|Reward = -(Free Energy) - (Number of Collision * Collision Punishment) - (Number of Trap * Trap Punishment)|

		Parameter : Bool, if 'Done = False' then reward = 0 (Sparse Reward = Reward Calculated At The End Of The Episode)
						  if 'Done = True' then calculate above reward formula
		Return    : list, (eg: ['H', 'H', 'P', ... 'H or P'])
	"""	

	def reward_function(self, Done = False):
		if Done:
			Collision = -2 * self.collision_punishment
			Trap = -4 * self.trap_punishment
			reward = -(self.free_energy) + Collision + Trap
		else:
			reward = 0

		return reward

	"""
	Codes Below Use To Calculate Total Free Energy Given Single Amino Coordinate
		Parameter :
		- new_coordinate_x = int, amino coordinate x axis
		- new_coordinate_y = int, amino coordinate y axis
		Return    : 
		- int, Free Energy  
	"""
	def energy_function(self, new_coordinat_x, new_coordinat_y):

		free_energy = 0
		energy, _ = self.check_Above_Neighbour(new_coordinat_x, new_coordinat_y)
		free_energy += energy
		energy, _ = self.check_Left_Neighbour(new_coordinat_x, new_coordinat_y)
		free_energy += energy
		energy, _ = self.check_Right_Neighbour(new_coordinat_x, new_coordinat_y)
		free_energy += energy
		energy, _ = self.check_Below_Neighbour(new_coordinat_x, new_coordinat_y)
		free_energy += energy
		
		return free_energy

	def update_action(self, action):
		if action == 0:
			action = 1
		elif action == 1:
			action = 2
		elif action == 2:
			action = 3
		elif action == 3:
			action = 0
		return action

	"""
	Codes Below Use To Check, If the particular amino had been trapped
		Parameter  : 
		- Done   = Bool, if 'Done = True' return Trap = False, otherwise check if it is trapped
		- move   = Int, action from action_space [0, 1, 2, 3]
		Return     : 
		-Trapped = Bool, True if amino trapped, False if it isnt
	"""

	def check_trapped(self,Done, move = None):
		#ignore if all amino had been drawn
		if Done: 
			trapped = False
			return trapped

		trapped = False

		if move == 0:
			_, amino_above = self.check_Above_Neighbour(self.prev_amino_position_x, self.prev_amino_position_y - self.amino_move)
			_, amino_left = self.check_Left_Neighbour(self.prev_amino_position_x, self.prev_amino_position_y - self.amino_move)
			_, amino_right = self.check_Right_Neighbour(self.prev_amino_position_x, self.prev_amino_position_y - self.amino_move)	
			_, amino_below = self.check_Below_Neighbour(self.prev_amino_position_x, self.prev_amino_position_y - self.amino_move)
		elif move == 1:
			_, amino_above = self.check_Above_Neighbour(self.prev_amino_position_x - self.amino_move, self.prev_amino_position_y)
			_, amino_left = self.check_Left_Neighbour(self.prev_amino_position_x - self.amino_move, self.prev_amino_position_y)
			_, amino_right = self.check_Right_Neighbour(self.prev_amino_position_x - self.amino_move, self.prev_amino_position_y)	
			_, amino_below = self.check_Below_Neighbour(self.prev_amino_position_x - self.amino_move, self.prev_amino_position_y)
		elif move == 2:
			_, amino_above = self.check_Above_Neighbour(self.prev_amino_position_x + self.amino_move, self.prev_amino_position_y)
			_, amino_left = self.check_Left_Neighbour(self.prev_amino_position_x + self.amino_move, self.prev_amino_position_y)
			_, amino_right = self.check_Right_Neighbour(self.prev_amino_position_x + self.amino_move, self.prev_amino_position_y)	
			_, amino_below = self.check_Below_Neighbour(self.prev_amino_position_x + self.amino_move, self.prev_amino_position_y)
		elif move == 3:
			_, amino_above = self.check_Above_Neighbour(self.prev_amino_position_x, self.prev_amino_position_y + self.amino_move)
			_, amino_left = self.check_Left_Neighbour(self.prev_amino_position_x, self.prev_amino_position_y + self.amino_move)
			_, amino_right = self.check_Right_Neighbour(self.prev_amino_position_x, self.prev_amino_position_y + self.amino_move)	
			_, amino_below = self.check_Below_Neighbour(self.prev_amino_position_x, self.prev_amino_position_y + self.amino_move)
		
		if amino_above == True and amino_below == True and amino_left == True and amino_right == True:
			trapped = True
		else:
			trapped = False

		return trapped

	"""
	Codes Below Use To Check Amino Collision
		Parameter  : 
		- move   = Int, action from action_space [0, 1, 2, 3]
		Return     : 
		-collide = Bool, True if amino collide, False if it isnt
	"""

	def check_collide(self, move = None):

		collide = False		

		#before taking action 0(up), check if it actually may collide 
		if move == 0:
			_, amino_above = self.check_Above_Neighbour(self.prev_amino_position_x, self.prev_amino_position_y)
			if amino_above == True:
				collide = True

		#check left	
		elif move == 1:
			_, amino_left = self.check_Left_Neighbour(self.prev_amino_position_x, self.prev_amino_position_y)
			if amino_left == True:
				collide = True

		#check right
		elif move == 2:
			_, amino_right = self.check_Right_Neighbour(self.prev_amino_position_x, self.prev_amino_position_y)
			if amino_right == True:
				collide = True

		#check below
		elif move == 3:
			_, amino_below = self.check_Below_Neighbour(self.prev_amino_position_x, self.prev_amino_position_y)
			if amino_below == True:
				collide = True
		
		else:
			collide = False


		return collide	
	
	#return readable RGB image for cv2	

	def get_image(self):
		
		image_data = Image.fromarray(self.current_image, 'RGB')
		return image_data	

	"""
	Codes Below Follows OpenAI Gym Function behaviour, such as 
	- reset()   : reset the environment to initial state (specify the amino_input is optional)
	- step()    : return new_state/observation, reward, done 
	- render()  : visualize to see whats going on
	"""

	def reset(self, amino_input = ['Nope']):

		#Initialize amino coordinate position for both x and y 
		self.prev_amino_position_x = self.init_amino_position_x
		self.prev_amino_position_y = self.init_amino_position_y

		#Reset Everything to Initial State
		self.init = True 
		self.Isreset = True
		
		self.collision_End = False
		self.collision_punishment = 0
		self.trap_punishment = 0
		self.free_energy = 0

		if amino_input[0] == 'Nope':
			self.amino_acid = self.generate_Random_Amino() #if there's no given input for amino_data, then generate a random sequence
		else:
			self.amino_acid = amino_input #otherwise, use the given input 

		self.amino_acid = self.preprocess_data(self.amino_acid) #preprocess amino data from string to int
		
		self.amino_size = self.adjust_amino_size() #adjust amino size, depend on amino_data length
		self.line_length = int(self.amino_size) #adjust line length, depend on amino_size and amino_data length
		self.amino_move = int(self.amino_size + self.line_length * 2) #movement distance of every step

		self.init_amino_data = np.zeros(100) #initialize 100 sequence with value zeros as default
		self.init_amino_data[:len(self.amino_acid)] = self.amino_acid #replace the init_amino_data based on given amino_data above
		self.amino_acid = self.init_amino_data #assign amino_data to amino_acid

		self.current_image = np.load("Background_Img/background.npy") #load background image
		
		current_small_image = cv2.resize(np.array(self.current_image), (150,150)) #resize image, because its nonsense to process 6000 by 6000 RGB image

		self.current_state = [self.amino_acid, current_small_image] #assign current_state 

		return self.current_state		

	def step(self, action):
		#init collide and trapped True, so it could do the 'collide' and 'trapped' checking process
		collide = True
		trapped = True

		if not self.Isreset:
			raise Exception("Cannot call env.step() before calling reset()")

		current_small_image = cv2.resize(np.array(self.current_image), (150,150)) #for new_state
		new_state = [self.amino_acid, current_small_image] #initialize 'new_state', just in case the 'if statement' below occur and it could return the 'new_state'.
		
		#if no amino acid left(all zeros), then return	new_state, reward, done
		if self.amino_acid[0] == 0:
			Done = True
			reward = 0
			return new_state, reward, Done 
		#Check if index - 1 is 0, which mean if its 'True' then now we are drawing the last amino acid. Return 'Done = True' if it does.
		elif self.amino_acid[1] == 0:
			Done = True		
		else:
			Done = False
		
		current_amino = self.amino_acid[0]	#current_amino always at index - 0
		
		# if init equals to True, then draw amino at the center of the image
		if self.init:
			if current_amino == 1.0:
				image_data = self.draw_amino(amino_type = 1, coordinat_x = self.init_amino_position_x, coordinat_y = self.init_amino_position_x, size = self.amino_size)
			else:
				image_data = self.draw_amino(amino_type = 2, coordinat_x = self.init_amino_position_x, coordinat_y = self.init_amino_position_y, size = self.amino_size)

			#To Update the amino_acid data, delete the amino after it had been drawn, also append another '0' value so it maintain the amino_data array size to 100.
			self.amino_acid = np.delete(self.amino_acid, 0)
			self.amino_acid = np.append(self.amino_acid, 0)
			
			self.init = False #set init to false, after done initialize the first amino

		current_amino = self.amino_acid[0]	#current amino always at index - 0

		#initialize 'collision' and 'trap' to 0, and increase it as it keeps collide
		collision = 0
		trap = 0
		while collide:
			collide = self.check_collide(move = action)
			if collide:
				collision += 1 

				if collision > 10: #nowhere to go, update_action didnt help. Then calculate the reward_function and Return :(
					Done = True					
					self.collision_End = True
					self.collision_punishment += collision
					self.trap_punishment += trap 	
					reward = self.reward_function(Done = True)					
					return new_state, reward, Done

				new_action = self.update_action(action)
				action = new_action
			
			trapped = self.check_trapped(Done, move = action)

			while trapped:
				trapped = self.check_trapped(Done, move = action)
				if trapped:
					trap += 1

					if trap > 10: #nowhere to go, update_action didnt help. Then calculate the reward_function and Return :(
						self.collision_End = True
						self.collision_punishment += collision
						self.trap_punishment += trap 
						reward = self.reward_function(Done = True)					
						return new_state, reward, Done

					new_action = self.update_action(action)
					action = new_action

			collide = self.check_collide(move = action)

		#sum up all the punishment value
		self.collision_punishment += collision 
		self.trap_punishment += trap 

		#Go Up
		if action == self.action_space[0]:

			if current_amino == 1:
				new_amino_position_x, new_amino_position_y, image_data = self.draw_next_amino(amino_type = 1, prev_coordinat_x = self.prev_amino_position_x, prev_coordinat_y = self.prev_amino_position_y, size = self.amino_size, action = 0)
				self.free_energy += self.energy_function(new_amino_position_x, new_amino_position_y)
			else:
				new_amino_position_x, new_amino_position_y, image_data = self.draw_next_amino(amino_type = 2, prev_coordinat_x = self.prev_amino_position_x, prev_coordinat_y = self.prev_amino_position_y, size = self.amino_size, action = 0)
				self.free_energy += 0

		#Go Left
		elif action == self.action_space[1]:

			if current_amino == 1:
				new_amino_position_x, new_amino_position_y, image_data = self.draw_next_amino(amino_type = 1, prev_coordinat_x = self.prev_amino_position_x, prev_coordinat_y = self.prev_amino_position_y, size = self.amino_size, action = 1)
				self.free_energy += self.energy_function(new_amino_position_x, new_amino_position_y)
			else:
				new_amino_position_x, new_amino_position_y, image_data = self.draw_next_amino(amino_type = 2, prev_coordinat_x = self.prev_amino_position_x, prev_coordinat_y = self.prev_amino_position_y, size = self.amino_size, action = 1)
				self.free_energy += 0

		#Go Right
		elif action == self.action_space[2]:

			if current_amino == 1:
				new_amino_position_x, new_amino_position_y, image_data = self.draw_next_amino(amino_type = 1, prev_coordinat_x = self.prev_amino_position_x, prev_coordinat_y = self.prev_amino_position_y, size = self.amino_size, action = 2)
				self.free_energy += self.energy_function(new_amino_position_x, new_amino_position_y)
			else:
				new_amino_position_x, new_amino_position_y, image_data = self.draw_next_amino(amino_type = 2, prev_coordinat_x = self.prev_amino_position_x, prev_coordinat_y = self.prev_amino_position_y, size = self.amino_size, action = 2)
				self.free_energy += 0

		#Go Down
		elif action == self.action_space[3]:

			if current_amino == 1:
				new_amino_position_x, new_amino_position_y, image_data = self.draw_next_amino(amino_type = 1, prev_coordinat_x = self.prev_amino_position_x, prev_coordinat_y = self.prev_amino_position_y, size = self.amino_size, action = 3)
				self.free_energy += self.energy_function(new_amino_position_x, new_amino_position_y)
			else:
				new_amino_position_x, new_amino_position_y, image_data = self.draw_next_amino(amino_type = 2, prev_coordinat_x = self.prev_amino_position_x, prev_coordinat_y = self.prev_amino_position_y, size = self.amino_size, action = 3)
				self.free_energy += 0

		if Done:
			reward = self.reward_function(Done = True) 

		else:
			reward = self.reward_function(Done = False)

		#update the previous postion to the new position
		self.prev_amino_position_x = new_amino_position_x
		self.prev_amino_position_y = new_amino_position_y
		#update current image to new image	
		self.current_image = image_data
		#as usual, resize current image
		current_small_image = cv2.resize(np.array(self.current_image), (150,150))
		#as usual 'delete and append' to update the amino_acid data
		self.amino_acid = np.delete(self.amino_acid, 0)
		self.amino_acid = np.append(self.amino_acid, 0)		

		new_state = [self.amino_acid, current_small_image] 

		return new_state, reward, Done
		
	def render(self, plot = False):
		"""
		Parameter : Bool, if 'plot = False' use to show the folding process, and show the matplotlib figure subsequently
						  if 'plot = True' then show only the matplotlib figure

		"""
		img = self.get_image()

		if plot:							
			plt.figure(figsize=(10,5))
			im = plt.imshow(img, interpolation='none')
			patches = []
			patches.append(mpatches.Patch(color = 'red', label="Free Energy = {}".format(self.free_energy)))
			patches.append(mpatches.Patch(color = 'blue', label="Collision = {}".format(self.collision_punishment)))
			patches.append(mpatches.Patch(color = 'yellow', label="Traps = {}".format(self.trap_punishment)))
			plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )					
			plt.show()
		else:
			cv2.namedWindow("output", cv2.WINDOW_NORMAL)
			cv2.resizeWindow('output', 900,900)
			cv2.imshow("output", np.array(img))	
			if self.amino_acid[0] == 0 or self.collision_End == True:
				if cv2.waitKey(0) & 0xFF == ord("q"):
					pass
			else:
				if cv2.waitKey(500) & 0xFF == ord("q"):
					pass
			if cv2.getWindowProperty('output', 0) < 0 and (self.amino_acid[0] == 0 or self.collision_End == True):
				plt.figure(figsize=(10,5))
				im = plt.imshow(img, interpolation='none')
				patches = []
				patches.append(mpatches.Patch(color = 'red', label="Free Energy = {}".format(self.free_energy)))
				patches.append(mpatches.Patch(color = 'blue', label="Collision = {}".format(self.collision_punishment)))
				patches.append(mpatches.Patch(color = 'yellow', label="Traps = {}".format(self.trap_punishment)))
				plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )		
				plt.show()
