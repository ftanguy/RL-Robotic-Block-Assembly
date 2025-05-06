#!/usr/bin/env python3
"""
Run a trained block‑assembly policy.

Usage example
-------------
python run_policy.py --model runs/bridge_maskppo_20250506_113000/final_model.zip \
                     --task bridge --algo maskppo --render
"""

import argparse
import logging
import time

from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

# ------------------------------------------------------------------------
# project imports (same as train.py)
# ------------------------------------------------------------------------
from assembly_gym import BlockAssemblyGym
from utils.logger_utils import get_logger

MAX_ACTIONS = 300
ALGOS = {
    "ppo": PPO,
    "maskppo": MaskablePPO,
}

# ------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------
def main():
    args = parse_args()

    logger = get_logger(__name__)
    level = logging.INFO

    if args.debug:
        level = logging.DEBUG
        logger.setLevel(level)
        logger.debug("Debug logging is enabled.")

    # ------------------------------------------------------------------
    # environment (single instance, render option)
    # ------------------------------------------------------------------
    base_env = BlockAssemblyGym(task=args.task,
                                level=level,
                                render=args.render)

    if args.algo == "maskppo":
        env = ActionMasker(base_env, lambda e: e.get_action_mask())
    else:
        env = base_env

    # ------------------------------------------------------------------
    # load model
    # ------------------------------------------------------------------
    Algo = ALGOS[args.algo]
    logger.info(f"Loading model from {args.model}")
    model = Algo.load(args.model, device=args.device)

    # ------------------------------------------------------------------
    # run one episode
    # ------------------------------------------------------------------
    obs, _ = env.reset(seed=args.seed)
    done, truncated = False, False
    total_reward = 0.0
    steps = 0
    while not (done or truncated):
        logger.debug(f"-------------- STEP : {steps} --------------")
        if args.algo == "maskppo":
            action_masks = env.action_masks()
            action, _ = model.predict(obs, deterministic=False, action_masks=action_masks)
        else:
            action, _ = model.predict(obs, deterministic=True)

        available_actions = base_env.backend.available_actions()
        #print_available_actions(available_actions)
        a = base_env.get_action_by_index(available_actions, action)
        logger.debug(f"\nSelected action {action}: {a}")

        obs, reward, done, truncated, _info = env.step(action)

        total_reward += reward
        steps += 1
        logger.debug("---------------------------------------------")
        time.sleep(1)

    logger.info(f"Episode finished after {steps} steps — total reward: {total_reward:.3f}")
    time.sleep(100)
    if args.render:
        logger.info("Close the matplotlib window to exit.")


# ------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Run a trained block‑assembly policy")
    p.add_argument("--model", required=True, help="Path to saved .zip model")
    p.add_argument("--task", choices=["bridge", "tower", "double_bridge"], default="bridge")
    p.add_argument("--num-stories", type=int, default=2)
    p.add_argument("--algo", choices=list(ALGOS.keys()), default="maskppo")
    p.add_argument("--device", default="auto")
    p.add_argument("--render", action="store_true", help="Render environment while running")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--debug", action="store_true")
    return p.parse_args()

# ------------------------------------------------------------------------
if __name__ == "__main__":
    main()
