import numpy as np
import tensorflow as tf
import random
import dqn
import gym
from collections import deque

env = gym.make('CartPole-v0')

input_size = env.observation_space.shape[0] # 4
output_size = env.action_space.n # 2

dis = 0.99
REPLAY_MEMORY = 50000 # 버퍼의 최대용량

def replay_train(DQN, train_batch):
	"""
	여기서 train_batch는 minibatch에서 가져온 data들입니다.
	x_stack은 state들을 쌓는 용도로이고,
	y_stack은 deterministic Q-learning 값을 쌓기 위한 용도입니다.
	우선 쌓기전에 비어있는 배열로 만들어놓기로 하죠.
	"""
	x_stack = np.empty(0).reshape(0, DQN.input_size) # array(10, 4)
	y_stack = np.empty(0).reshape(0, DQN.output_size) # array(10, 2)

	# Get stored information from the buffer
	"""for를 통해서 minibatch(train_batch)에서 가져온 값들을 하나씩 꺼냅니다."""
	for state, action, reward, next_state, done in train_batch:
		Q = DQN.predict(state)

		# terminal
		if done:
			Q[0, action] = reward
		else :
			# Obtain the Q' values by feeding the new state through our network
			Q[0, action] = reward + dis * np.max(DQN.predict(next_state))

		y_stack = np.vstack([y_stack, Q])
		x_stack = np.vstack([x_stack, state])

	# Train our network using target and predicted Q values on each episode
	"""
	쌓은 stack들을 바로 update로 돌려서 학습을 시킵니다.
	학습은 위에서 만들었던 neural network(linear regression)을 통해서 학습이 되겠지요.
	"""
	return DQN.update(x_stack, y_stack)

def bot_play(mainDQN) :
	"""
	실제로 학습된 것으로 돌려보는 코드입니다.
	mainDQN 역시 학습된 network입니다.
	"""
	# See our trained network in action
	s = env.reset()
	reward_sum = 0

	while True:
		env.render()
		a = np.argmax(mainDQN.predict(s))
		s, reward, done, _ = env.step(a)
		reward_sum += reward
		if done:
			print("Total score: {}".format(reward_sum))
			break

def main():
	max_episodes = 5000

	# store the previous observations in replay memory
	"""python에 내장 되어있는 deque를 이용하여 buffer를 만듭니다."""
	replay_buffer = deque()

	with tf.Session() as sess :
		"""
		dqn.py를 통해서 DQN 클래스에 있는 network만든 것을 생성합니다.
		"""
		mainDQN = dqn.DQN(sess, input_size, output_size)
		tf.global_variables_initializer().run()

		for episode in range(max_episodes):
			e = 1. / ((episode / 10) + 1)
			done = False
			step_count = 0
			"""처음에 0으로 초기화!"""
			state = env.reset()

			while not done:
				# E-greedy
				if np.random.rand(1) < e :
					action = env.action_space.sample()
				else:
					# Choose an action by greedilty from the Q-network
					action = np.argmax(mainDQN.predict(state))

				# Get new state and reward from environment
				next_state, reward, done, _ = env.step(action)
				if done: # big penalty
					reward = -1

				# Save the experience to our buffer
				"""
				각 state마다 env에서 action을 한 값들
				(state, action, reward, next_state, done)을 버퍼에 저장하는 코드입니다.
				"""
				replay_buffer.append((state, action, reward, next_state, done))

				"""
				만약에 버퍼에 저장한 값이 너무 많으면 안되니까 REPLAY_MEMORY값을 넘으면
				맨 왼쪽에 있던 값을 빼버리는(pop) 코드입니다.(스택은 비커, 큐는 식도)
				"""
				if len(replay_buffer) > REPLAY_MEMORY: # REPLAY_MEMORY = 50000
					replay_buffer.popleft()

				state = next_state
				step_count += 1

			print("Episode: {} step: {}".format(episode, step_count))

			if episode % 10 == 1 : # train every 10 episodes
			# """
			# max_episodes = 5000, for episode in range(max_episodes)
			# 일정 시간이 지나면, 일정 episode가 쌓이면 학습을 시켜야합니다.
			# """
				# Get a random batch of experiences.
				for _ in range(50):
					# Minibatch works better
					"""버퍼에서 10개를 랜덤으로 가져옵니다. -> minibatch"""
					minibatch = random.sample(replay_buffer, 10)
					loss, _ = replay_train(mainDQN, minibatch)
					"""
					simple_replay_train를 통해 학습을 시킵니다.
					simple_replay_train함수로 가볼까요?
					"""
				print("Loss: ", loss)

		bot_play(mainDQN)
		"""학습이 된 것으로 게임을 돌려봅시다~!"""

if __name__ == "__main__":
	main()