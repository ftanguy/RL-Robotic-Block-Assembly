import matplotlib.pyplot as plt
import numpy as np
from rendering import plot_assembly_env, plot_task
# from tree import ExtendedTree, Action
import tasks as ts
from assembly_env import AssemblyEnv
from blocks import Floor

task = ts.Bridge(num_stories=3)
env = AssemblyEnv(task)
done = False
rewards = 0

# Initialize figure for plotting
plt.ion()
fig, ax = plt.subplots()

while not done:
    action = env.random_action(non_colliding=True, stable=False)
    if action is None:
        break

    print(f"Action: {action}")
    obs, r, done = env.step(action)

    # Convert observation to NumPy array if needed
    obs_np = np.array(obs)

    # Print observation summary
    print(f"obs shape: {obs_np.shape}")
    print(f"obs sum: {obs_np.sum()}")
    unique_vals = np.unique(obs_np)
    print(f"unique values in obs: {unique_vals}")

    # Plot the observation
    ax.clear()
    im = ax.imshow(obs_np[0], cmap='gray', interpolation='none')
    ax.set_title("Observation")
    plt.pause(1)

    rewards += r
    print(f"Environment stable: {env.is_stable()}")

print("Total rewards:", rewards)

# Final environment plot
plt.ioff()
plot_assembly_env(env, task=task, plot_forces=False)
plt.axis('equal')
plt.show()
