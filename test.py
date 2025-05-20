import torch
from assembly_gym import BAActorCritic, BlockAssemblyGym
from sb3_contrib.common.wrappers import ActionMasker
import numpy as np

env = BlockAssemblyGym("bridge")
obs, _ = env.reset()
assert obs["images"].shape == (2, 64, 64)
assert obs["actions"].shape == (512, 33)
assert env.action_masks().shape == (512,)

acts = env.backend.available_actions()
mask = env.action_masks()
assert mask.sum() == len(acts)
for i,m in enumerate(mask):          # ensure no stray bits
    assert bool(m) == (i < len(acts))

idx = np.where(mask)[0][-1]          # last legal
obs2, r, done, trunc, _ = env.step(idx)
assert not np.isnan(r)

policy = BAActorCritic(env.observation_space, env.action_space, lr_schedule=lambda _:0.001)
am_env  = ActionMasker(env, lambda e: e.action_masks())
obs, _  = am_env.reset()
obs_t   = policy.obs_to_tensor(obs)[0]
act, val, logp = policy.forward(obs_t, action_masks=am_env.action_masks())
assert act.shape == (1,)
assert val.shape == (1,1)
assert not torch.isnan(logp).any()