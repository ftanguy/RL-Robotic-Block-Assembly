import matplotlib.pyplot as plt
from rendering import plot_assembly_env, plot_task
#from tree import  ExtendedTree, Action
import tasks as ts
from assembly_env import AssemblyEnv
from blocks import Floor


task = ts.Bridge(num_stories=3)

env = AssemblyEnv(task)
done = False
rewards = 0

while not done:
    action = env.random_action(non_colliding=True, stable=False)
    print(action)
    print(env.encode_action(action))
    print(env.encode_action(action).shape)

    obs, r, done = env.step(action)

    rewards += r

print(rewards)

plot_assembly_env(env, task=task,plot_forces=False)
plt.axis('equal')
plt.show()