import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


environment = gym.make("CartPole-v0")

inputSize = environment.observation_space.shape[0]     # 4
outputSize = environment.action_space.n         # 2
learningRate = 0.1

input = tf.placeholder(shape=[1, inputSize], dtype=tf.float32)
weight = tf.get_variable("W1", shape= [inputSize, outputSize],
                         initializer= tf.contrib.layers.xavier_initializer())
Qpred = tf.matmul(input, weight)

label = tf.placeholder(shape=[None, outputSize], dtype=tf.float32)

error = tf.reduce_sum(tf.square(tf.subtract(label, Qpred)))
train = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(error)

discount = 0.99
trainNumber = 2001
rewardList = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(trainNumber):
        state = environment.reset()
        boundary = 1.0 / ((i / 50) + 10)
        totalReward = 0
        isDone = False

        while not isDone:
            x = np.reshape(state, [1, inputSize])
            Q = sess.run(Qpred, feed_dict={input: x})

            if np.random.rand(1) < boundary:
                action = environment.action_space.sample()
            else:
                action = np.argmax(Q)

            nextState, reward, isDone, probability = environment.step(action)

            if isDone:
                Q[0, action] = -100
            else:
                x1 = np.reshape(nextState, [1, inputSize])
                Qs1 = sess.run(Qpred, feed_dict={input: x1})
                Q[0, action] = reward + discount * np.max(Qs1)

            sess.run(train, feed_dict={input: x, label: Q})
            totalReward += reward
            state = nextState

        rewardList.append(totalReward)
        print("Episode : {}, steps : {}".format(i, totalReward))
        if len(rewardList) > 10 and np.mean(rewardList[-10:]) > 500:
            break

    observation = environment.reset()
    totalReward = 0
    while True:
        environment.render()
        x=np.reshape(observation, [1, inputSize])
        Q = sess.run(Qpred, feed_dict={input : x})
        a = np.argmax(Q)

        observation, reward, isDone, _ = environment.step(a)
        totalReward +=reward
        if isDone:
            print("Total score : {}".format(totalReward))
            break
