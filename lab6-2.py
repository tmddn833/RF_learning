import gym

environment = gym.make('CartPole-v0')
environment.reset()

randomNumber = 10
totalReward = 0

while randomNumber > 0:
    environment.render()
    action = environment.action_space.sample()

    nextState, reward, isDone, probability = environment.step(action)
    totalReward += reward

    print(" state {}  reward {}  isDone {}".format(nextState, reward, isDone), end='')
    print()

    if isDone:
        randomNumber -= 1
        print("====================")
        print("total reward {}".format(totalReward))
        print("====================")

        totalReward = 0
        environment.reset()