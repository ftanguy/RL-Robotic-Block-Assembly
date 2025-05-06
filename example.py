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
    if action is None:
        break
    print(action)
    obs, r, done = env.step(action)
    print(f"obs: {obs}")
    print(f"obs shape: {obs.size()}")
    print(f"obs sum: {obs.sum()}")
    rewards += r
    print(env.is_stable())
    
print(rewards)

plot_assembly_env(env, task=task,plot_forces=False)
plt.axis('equal')
plt.show()