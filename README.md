## How to setup the environment

    ''' bash
    conda env create -f block_rl_env.yml
    '''

    To use stable baseline you might need to install the extras with this command

    pip install stable-baselines3\[extra\]

## How does the env work?

### Assembly_env.py
    This file creates the environment for the block stacking

### Assembly_gym.py
    This file creates a gym wrapper around the assembly_env making it possible to use it with RL frameworks such as stable-baseline3

### train.py
    This file trains an agent using the assembly_gym env.

    These are the launch options

    Here is an example on how to run a trainig
    ...

    python train.py --task bridge --algo maskppo --timesteps 100000 --device cpu --progress-bar --config configs/maskppo.yaml  


    Here is an example on how to continue the training of a previously trained agent

    python train.py --task bridge --algo maskppo --timesteps 100000 --device cpu --progress-bar --config configs/maskppo.yaml  -m runs/bridge_maskppo_0506204539/best_model/best_model.zip

    #### configs/maskppo.yaml
    This contains the hyperparameters for the maskppo RL algorithm which trains the agent on the given task

### run_policy.py
    This file allows you to run a trainined policy to see results

    Here are the launch options:
    ...

    Here is an example on how to test a trained policy:

    python run_policy.py --model runs/bridge_maskppo_0506212053/best_model/best_model.zip --task bridge --algo maskppo --render --debug


## RL choices

### Maskedppo
    Maskedppo
    observations
    action space
    reward function

### Areas to improve/implement

# Block‑Assembly RL

---

## How to setup the environment

```bash
# create the Conda environment 
$ conda env create -f block_rl_env.yml
$ conda activate block_rl

# Stable‑Baselines3 extras: usefull to have the training progress bar.
$ pip install stable-baselines3\[extra\]

```

---

## How does the env work?

### `assembly_env.py`

Low‑level **geometry & physics** backend.  Responsible for:

* keeping the list of blocks (`self.block_list`)
* collision checks & static stability (`is_stable_rbe`)
* dense reward heat‑map generation

### `assembly_gym.py`

Gymnasium **wrapper** that the RL agent actually interacts with.  It:

* exposes a `Discrete(300)` action space with automatic **action‑masking**
* concatenates *state image* + *reward image* into a single flat observation
* offers live Matplotlib rendering (`--render`)

### `train.py`

Train an agent with **Stable‑Baselines3**.  Key CLI flags (run `-h` for all):

| Flag                   | Default     | Description                                               |
| ---------------------- | ----------- | --------------------------------------------------------- |
| `--task`               | `bridge`    | Task to learn: `bridge`, `tower`, `double_bridge`         |
| `--algo`               | `maskppo`   | RL algorithm: `maskppo` (masked PPO) or plain `ppo`       |
| `--timesteps`          | `200_000`   | Total training steps (across **all** envs)                |
| `--save-freq`          | `10_000`    | Checkpoint frequency (steps) for saving models & eval     |
| `--logdir`             | `runs`      | Output directory for checkpoints and TensorBoard logs     |
| `--device`             | `cpu`       | Compute device: `cpu`, `cuda`, or `auto`                  |
| `--render`             | `False`     | Render the environment (only works when `--n-envs 1`)     |
| `--debug`              | `False`     | Enable DEBUG‑level logging                                |
| `--progress-bar`       | `False`     | Show SB3 progress bar during training                     |
| `--config`             | *None*      | Path to YAML with extra hyper‑parameters (overrides CLI)  |
| `-m`, `--resume-model` | *None*      | Path to a `.zip` model to continue training from          |
| `-n`, `--n-envs`       | `1`         | Number of parallel environments (≥2 uses `SubprocVecEnv`) |

**Train from scratch**

```bash
python train.py --task bridge --algo maskppo --timesteps 100000 --progress-bar --config configs/maskppo.yaml  
```

**Resume training**

```bash
python train.py --task bridge --algo maskppo --timesteps 100000 --progress-bar --config configs/maskppo.yaml  -m runs/bridge_maskppo_0506204539/best_model/best_model.zip
```

### `run_policy.py`

Roll out a **trained policy** for qualitative inspection.

```bash
python run_policy.py --model runs/bridge_maskppo_0506212053/best_model/best_model.zip --task bridge --algo maskppo --render --debug
```

Here is an example of a rollout

![](Images/001.gif)
![](Images/002.gif)

---

## RL choices

### Maskable PPO (`maskppo`)

* **Observation**   8192‑D vector (2 × 64 × 64 images flattened).
* **Action space**  300 discrete indices; 
* **Reward**        sum of overlaps between the newly placed block and Gaussian blobs centred on targets.
* **Masking**       `sb3_contrib.ActionMasker` removes illegal moves before softmax → faster learning & fewer crashes.

Other SB3 algorithms (SAC, A2C…) will work, but the policy network must be adapted to flat image inputs.

---

## Project Ideas to implement (at least 1)

- The policy should generate various kind of possible structures for a same task
- The policy should be able to complete an episode starting from some arbitrary/random situation not seen during training.
- Generate some interesting and novel structures
- Minimise the number of blocks used by the policy to complete the episode
- Use new types of block geometries
- Train a policy that is robust to noise injected when placing a block (important for sim2real transition to a real robot).

---


