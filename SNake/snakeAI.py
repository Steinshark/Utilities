import pygame
from random import randint
import random
import time
import pprint
import networks
import json
import numpy as np
import os
import torch
import torch.nn as nn
import copy

class SnakeGame:

	def __init__(self,w,h,fps=30):
		self.width = w
		self.height = h

		self.food = (randint(0,self.width - 1),randint(0,self.height - 1))
		self.snake = [[0,0]]

		self.colors = {"FOOD" : (255,20,20),"SNAKE" : (20,255,20)}

		self.frame_time = 1 / fps

		self.snapshot_vector = [[0 for x in range(self.height)] for i in range(self.width)]

		self.direction = (1,0)

		self.data = []

	def play_game(self,window_x,window_y,training_match=True,model=None):


		if not window_x == window_y:
			print(f"invalid game_size {window_x},{window_y}.\nDimensions must be equal")
			return

		square_width 	= window_x / self.width
		square_height 	= window_y / self.height


		#Display setup
		pygame.init()
		self.window = pygame.display.set_mode((window_x,window_y))
		pygame.display.set_caption("AI Training!")


		self.output_vector = [0,0,0,1]
		game_running = True

		while game_running:
			self.window.fill((0,0,0))
			pygame.event.pump()
			t_start = time.time()
			keys = pygame.key.get_pressed()


			#print(keys[pygame.K_w])
			f_time = t_start - time.time()

			#Draw snake and food
			if training_match:
				self.update_movement()
				self.create_input_vector()

			else:
				assert model is not None

				y_feed = torch.tensor(self.game_to_model(self.create_input_vector()),dtype=torch.float)
				model_out = model.forward(y_feed)

				w,s,a,d = model_out.cpu().detach().numpy()
				print([w,a,s,d])
				keys= pygame.key.get_pressed()
				if True in [keys[pygame.K_w],keys[pygame.K_a],keys[pygame.K_s],keys[pygame.K_d]]:
					print("overriding ML")
					self.update_movement(player_input=True)
				else:
					self.update_movement(player_input=False,w=w,s=s,a=a,d=d)


			for coord in self.snake:
				x,y = coord[0] * square_width,coord[1] * square_height
				new_rect = pygame.draw.rect(self.window,self.colors["SNAKE"],pygame.Rect(x,y,square_width,square_height))
			x,y = self.food[0] * square_width,self.food[1] * square_height
			food_rect = pygame.draw.rect(self.window,self.colors["FOOD"],pygame.Rect(x,y,square_width,square_height))
			pygame.display.update()


			#Movement
			next_x = self.snake[0][0] + self.direction[0]
			next_y = self.snake[0][1] + self.direction[1]
			if next_x >= self.width or next_y >= self.height or next_x < 0 or next_y < 0:
				game_running = False
			next_head = (next_x , next_y)

			if next_head in self.snake:
				print("you lose!")
				game_running = False

			if next_head == self.food:
				self.food = (randint(0,self.width - 1),randint(0,self.height - 1))
				self.snake = [next_head] + self.snake
			else:
				self.snake = [next_head] + self.snake[:-1]

			if keys[pygame.K_p]:
				print(f"input vect: {self.vector}")
				print(f"\n\noutput vect:{self.output_vector}")
			#Keep constant frametime
			self.data.append({"x":self.input_vector,"y":self.output_vector})
			if self.frame_time > f_time:
				time.sleep(self.frame_time - f_time)


		self.save_data()

	def save_data(self):
		x = []
		y = []
		for item in self.data[:-1]:
			x_item = np.ndarray.flatten(np.array(item["x"]))
			y_item = np.array(item["y"])

			x.append(x_item)
			y.append(y_item)

		x_item_final = np.ndarray.flatten(np.array(self.data[-1]["x"]))
		y_item_final = list(map(lambda x : x * -1,self.data[-1]["y"]))

		x.append(x_item_final)
		y.append(y_item_final)

		x = np.array(x)
		y = np.array(y)

		if not os.path.isdir("experiences"):
			os.mkdir("experiences")

		i = 0
		fname = f"exp_x_{i}.npy"
		while os.path.exists(os.path.join("experiences",fname)):
			i += 1
			fname = f"exp_x_{i}.npy"
		np.save(os.path.join("experiences",fname),x)
		np.save(os.path.join("experiences",f"exp_y_{i}.npy"),y)

	def game_to_model(self,x):
		return np.ndarray.flatten(np.array(x))

	def create_input_vector(self):
		self.input_vector = [[0 for x in range(self.height)] for y in range(self.width)]
		self.input_vector[self.snake[0][1]][self.snake[0][0]] = 1
		for piece in self.snake[1:]:
			self.input_vector[piece[1]][piece[0]] = -1
		food_placement = [[0 for x in range(self.height)] for y in range(self.width)]
		food_placement[self.food[1]][self.food[0]] = 1
		self.input_vector += food_placement
		return self.input_vector

	def update_movement(self,player_input=False,w=0,s=0,a=0,d=0):

		if player_input:
			pygame.event.pump()
			keys = pygame.key.get_pressed()
			w,s,a,d = (0,0,0,0)

			if keys[pygame.K_w]:
				w = 1
			elif keys[pygame.K_s]:
				s = 1
			elif keys[pygame.K_a]:
				a = 1
			elif keys[pygame.K_d]:
				d = 1
			else:
				return
			self.output_vector = [w,s,a,d]

		self.movement_choices = {
			(0,-1) 	: w,
			(0,1) 	: s,
			(-1,0) 	: a,
			(1,0)	: d}

		self.direction = max(self.movement_choices,key=self.movement_choices.get)

	def train_on_game(self,model,visible=True,epsilon=.05):
		window_x, window_y = (600,600)
		experiences = []
		rewards = {"die":-1,"food":1,"live":.05,"idle":-.5}

		#setup
		assert model is not None
		square_width 	= window_x / self.width
		square_height 	= window_y / self.height
		pygame.init()
		game_running = True
		output_vector = [0,0,0,1]
		eaten_since = 0
		#Game display
		if visible:
			self.window = pygame.display.set_mode((window_x,window_y))
			pygame.display.set_caption("AI Training!")

		#Game Loop
		while game_running:
			input_vector = self.get_state_vector()

			#Find next update_movement
			if random.random() < epsilon:
				x = random.randint(-1,1)
				y = random.randint(-1,1)
				while abs(x) == abs(y):
					x = random.randint(-1,1)
					y = random.randint(-1,1)
				self.direction = (x,y)
			else:
				movement_values = model.forward(input_vector)
				w,s,a,d = movement_values.cpu().detach().numpy()
				self.update_movement(w=w,s=s,a=a,d=d)

			#Game display
			if visible:
				self.window.fill((0,0,0))
				#print(f"model predicts: {[w,a,s,d]}")


			#Game Display
			if visible:
				for coord in self.snake:
					x,y = coord[0] * square_width,coord[1] * square_height
					new_rect = pygame.draw.rect(self.window,self.colors["SNAKE"],pygame.Rect(x,y,square_width,square_height))
				x,y = self.food[0] * square_width,self.food[1] * square_height
				food_rect = pygame.draw.rect(self.window,self.colors["FOOD"],pygame.Rect(x,y,square_width,square_height))
				pygame.display.update()


			#Game Logic
			next_x = self.snake[0][0] + self.direction[0]
			next_y = self.snake[0][1] + self.direction[1]
			next_head = (next_x , next_y)

			#Check lose
			if next_x >= self.width or next_y >= self.height or next_x < 0 or next_y < 0 or next_head in self.snake:
				game_running = False
				reward = rewards['die']
				experiences.append({'s':input_vector,'r':reward,'a':self.direction,'s`':'terminal'})
				return experiences

			#Check eat food
			eaten_since += 1
			if next_head == self.food:
				eaten_since = 0
				self.food = (randint(0,self.width - 1),randint(0,self.height - 1))
				self.snake = [next_head] + self.snake
				reward = rewards['food']
			else:
				self.snake = [next_head] + self.snake[:-1]
				reward = rewards['live']

			#Add to experiences
			experiences.append({'s':input_vector,'r':reward,'a':self.direction,'s`': self.get_state_vector()})

			if eaten_since > 2000:
				game_running = False
				reward = rewards['idle']
				experiences.append({'s':input_vector,'r':reward,'a':self.direction,'s`':'terminal'})
				return experiences
		return experiences

	def get_state_vector(self):
		#Build x by y vector for snake
		input_vector = [[0 for x in range(self.height)] for y in range(self.width)]

		#Head of snake == 1
		input_vector[self.snake[0][1]][self.snake[0][0]] = 1

		#Rest of snake == -1
		for piece in self.snake[1:]:
			input_vector[piece[1]][piece[0]] = -1

		#Build x by y vector for food placement
		food_placement = [[0 for x in range(self.height)] for y in range(self.width)]
		food_placement[self.food[1]][self.food[0]] = 1
		input_vector += food_placement

		#Translate to numpy and flatten
		np_array = np.ndarray.flatten(np.array(input_vector))
		#Translate to tensor
		return torch.tensor(np_array,dtype=torch.float)


class Trainer:

	def __init__(self,memory_size,visible=True,loading=True,PATH="E:\code\Scratch\models",fps=200):
		self.memory_size = memory_size
		self.experiences = []

		self.target_model 	= networks.FullyConnectedNetwork(800,4,5e-6)
		self.learning_model = networks.FullyConnectedNetwork(800,4,5e-6)
		if loading:
			self.target_model.load_state_dict(torch.load(os.path.join(PATH,"t_model")))
			self.learning_model.load_state_dict(torch.load(os.path.join(PATH,"l_model")))
			print("sucessfully loaded models")
			#self.target_model.eval()
			#self.learning_model.eval()
		self.visible = visible
		self.indices = [(0,-1),(0,1),(-1,0),(1,0)]
		self.gamma = .3
		self.PATH = PATH
		self.loss = nn.MSELoss()
		self.optimizer = torch.optim.Adam(self.learning_model.parameters())
		self.fps = fps
	def train(self,episodes=1000,replay_every=100,sample_size=5000,batch_size=8):

		for e_i in range(episodes):
			#Play a game and collect the experiences
			game = SnakeGame(20,20,fps=100)
			self.experiences += game.train_on_game(self.learning_model,visible=self.visible)

			#If training on this episode
			if e_i % replay_every == 0 and not e_i == 0:
				training_set = [self.experiences[random.randint(0,len(self.experiences)-1)] for _ in range(sample_size)]
				print(f"EPISODE {e_i}/{episodes} ----------------------------------")
				self.train_on_experiences(training_set,batch_size=batch_size)
				self.transfer_models()
				self.experiences = []
		torch.save(self.target_model.state_dict(),os.path.join(self.PATH,"t_model"))
		torch.save(self.learning_model.state_dict(),os.path.join(self.PATH,"l_model"))

	def train_on_experiences(self,big_set,epochs=1,batch_size=8):

		for epoch_i in range(epochs):
			batches = [[big_set[i * n] for n in range(batch_size)] for i in range(int(len(big_set)/batch_size))]
			print(f"\tepoch {epoch_i}")
			experiences = torch.from_numpy(np.array([exp['s'].cpu().detach().numpy() for exp in big_set]))

			c_loss = 0
			for batch,exp_set in enumerate(batches):
				indices = [exp['a'] for exp in exp_set]
				vals = self.learning_model(experiences)
				vals_target_adjusted = torch.clone(vals)

				for item,i in enumerate(indices):
					if exp_set[item]['s`'] == 'terminal':
						target = exp_set[item]['r']
					else:
						next_state_val = max(self.target_model(exp_set[item]['s`']))
						target = exp_set[item]['r'] + self.gamma * next_state_val
					vals_target_adjusted[item,i] = target

				self.learning_model.optimizer.zero_grad()
				loss = self.learning_model.loss(vals,vals_target_adjusted)
				c_loss += loss
				#Perform grad descent
				loss.backward()
				self.learning_model.optimizer.step()

				print(f"\tBatch {batch}/{len(batches)}:\t loss={c_loss/(batch+1):.8f}",end="\r")
			print("")

	def transfer_models(self):
		torch.save(self.learning_model.state_dict(),os.path.join(self.PATH,"l_model"))
		self.target_model.load_state_dict(torch.load(os.path.join(self.PATH,"l_model")))
if __name__ == "__main__":

	import sys
	if sys.argv[1] in ['-p',"--play"]:
		s = SnakeGame(20,20,fps=15)
		s.play_game(600,600,training_match=True)
		exit()

	elif sys.argv[1] in ['-r','--reinforce']:
		try:
			trainer = Trainer(100000,visible=sys.argv[2] in ["T",'t'],loading=sys.argv[3] in ["t","T"],PATH="models")
			trainer.train(episodes=int(sys.argv[4]),replay_every=int(sys.argv[5]),sample_size=int(sys.argv[6]),batch_size=int(sys.argv[7]))
		except ValueError as i:
			print(i)
			print(f"usage: --reinforce visible[t/f] loading[t/f] episodes[n] replay_every[n] sample_size[n]")
	else:
		exp = {}
		x_exp = {}
		y_exp = {}

		for f in os.listdir("experiences"):
			type = f[4]
			number = f[6:].split(".")[0]

			if not number in exp:
				exp[number] = {"x" :None,"y":None}

			if type == "x":
				exp[number]["x"] = np.load(os.path.join("experiences",f))
			else:
				exp[number]["y"] = np.load(os.path.join("experiences",f))

		final_boss_list = {"x":np.empty((0,800)),"y":np.empty((0,4))}
		for num in exp:
			try:
				final_boss_list["x"] = np.append(final_boss_list['x'], exp[num]["x"],axis=0)
				final_boss_list["y"] = np.append(final_boss_list['y'], exp[num]["y"],axis=0)
			except ValueError:
				print(f"{num} did not play happy\n\n")


		x_data = torch.from_numpy(final_boss_list["x"]).float()
		y_data = torch.from_numpy(final_boss_list["y"]).float()

		model = networks.FullyConnectedNetwork(800,4,1e-7)

		print(f"training model on {x_data.shape} datapoints")
		model.train(x_data,y_data,verbose=True,epochs=int(sys.argv[2]),show_steps=25,batch_size=int(sys.argv[3]),show_graph=sys.argv[4] in ["T",'t'])

		game = SnakeGame(20,20,fps=4)

		game.play_game(600,600,training_match=False,model=model)
