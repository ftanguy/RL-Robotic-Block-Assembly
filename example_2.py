import matplotlib.pyplot as plt
from rendering import plot_assembly_env, plot_task
from tasks import Bridge
from assembly_env import AssemblyEnv, Action
from blocks import Floor




task = Bridge(num_stories=2)

env = AssemblyEnv(task)

action = Action(target_block=0, target_face=0, shape=1, face = 0, offset_x = -1)
obs, r, done = env.step(action)

if 1:
    action = Action(target_block=1, target_face=3, shape=5, face = 2, offset_x = -0.)
    obs, r, done = env.step(action)
    plt.imshow(obs[0], cmap='gray', interpolation='none')
    plt.show()

    action = Action(target_block=0, target_face=0, shape=1, face = 0, offset_x = 1.5)
    obs, r, done = env.step(action)
    plt.imshow(obs[0], cmap='gray', interpolation='none')
    plt.show()

    action = Action(target_block=3, target_face=3, shape=5, face = 1, offset_x = 0.)
    obs, r, done = env.step(action)
    plt.imshow(obs[0], cmap='gray', interpolation='none')
    plt.show()

if 0:
    action = Action(target_block=0, target_face=0, shape=1, face = 0, offset_x = -1.5)
    obs, r, done = env.step(action)
    print(r)
    print(env.is_stable())
    action = Action(target_block=1, target_face=3, shape=5, face = 2, offset_x = 0.)
    obs, r, done = env.step(action)
    print(r)
    print(env.is_stable())
    action = Action(target_block=0, target_face=0, shape=1, face = 0, offset_x = 1.5)
    obs, r, done = env.step(action)
    print(r)
    print(env.is_stable())
    action = Action(target_block=3, target_face=3, shape=5, face = 1, offset_x = 0.)
    obs, r, done = env.step(action)
    print(r)
    print(env.is_stable())
    action = Action(target_block=4, target_face=2, shape=5, face = 1, offset_x = 0.)
    obs, r, done = env.step(action)
    print(r)
    print(env.is_stable())
    print(done)


plot_assembly_env(env, task=task)
plt.axis('equal')
plt.show()