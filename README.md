# RL Target-Driven Block Assembly

## Project Overview & Results
This project explores target-driven 2-D block assembly using Reinforcement Learning. The agent must sequentially place up to ten heterogeneous blocks to cover prescribed target pixels, avoid fixed obstacles, and keep the structure statically stable.

**The Challenge:** The environment presents a variable, high-cardinality action set where every step yields up to 512 collision-free placement candidates. The feasibility of these placements constantly changes with the evolving structure.

**Our Approach:** We adapted a convolutional-MLP policy trained with Maskable Proximal Policy Optimisation (Maskable PPO). The policy:
1. Encodes each candidate action with one-hot embeddings.
2. Applies a dynamic environment mask to suppress invalid logits.
3. Selects a placement in a single forward pass.

We also extended the baseline environment to remove collision-inducing actions before learning and to log dense per-block rewards.

**Results:** * The maskable policy reliably learns to avoid premature episode termination (instability or collisions). 
* The average episode length successfully stabilizes at the maximum limit of 10 steps.
* The agent attains a higher dense reward than an unmasked PPO baseline. 
* Qualitative analysis shows the agent successfully learns to build vertically toward targets, though it currently struggles to span horizontal gaps to build "bridges".

---

## How to setup the environment

```bash
# create the Conda environment 
$conda env create -f block_rl_env.yml$ conda activate block_rl

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
| `--config`             | *None* | Path to YAML with extra hyper‑parameters (overrides CLI)  |
| `-m`, `--resume-model` | *None* | Path to a `.zip` model to continue training from          |
| `-n`, `--n-envs`       | `1`         | Number of parallel environments (≥2 uses `SubprocVecEnv`) |

**Train from scratch**

```bash
python train.py --task bridge --algo maskppo --timesteps 100000 --progress-bar --config configs/maskppo.yaml  
```

**Resume training**

```bash
python train.py --task bridge --algo maskppo --timesteps 100000 --progress-bar --config configs/maskppo.yaml  -m runs/bridge_maskppo_0506204539/best_model/best_model.zip
```

**Monitor training**
To monitor the training you can run the following command in the main directory.

```bash
tensorboard --logdir runs
```

![](Images/training_plots.png)

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

* **Observation** 8192‑D vector (2 × 64 × 64 images flattened).
* **Action space** 300 discrete indices; 
* **Reward** sum of overlaps between the newly placed block and Gaussian blobs centred on targets.
* **Masking** `sb3_contrib.ActionMasker` removes illegal moves before softmax → faster learning & fewer crashes.

Other SB3 algorithms (SAC, A2C…) will work, but the policy network must be adapted to flat image inputs. 

---

## Contributors
* Tim Lücking
* Matthew Meyer
* Yuan Xiao
* Florian Tanguy
* Renqing Cuomao
* Alix Papadatos
