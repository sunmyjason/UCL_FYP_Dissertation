# Consolidated import statements
import sys
sys.path.append("../")  # Add repo path

# Standard Libraries
import os
import shutil
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import time
import optuna
import traceback
from multiprocessing import Pool, cpu_count

# Gym and Stable Baselines3
import gym
from stable_baselines3 import A2C, PPO, SAC
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

# MBT Gym imports
from mbt_gym.gym.wrappers import ReduceStateSizeWrapper
from mbt_gym.gym.StableBaselinesTradingEnvironment import StableBaselinesTradingEnvironment
from mbt_gym.gym.TradingEnvironment import TradingEnvironment
from mbt_gym.gym.helpers.generate_trajectory import generate_trajectory
from mbt_gym.rewards.RewardFunctions import PnL, CjMmCriterion
from mbt_gym.stochastic_processes.midprice_models import *
from mbt_gym.stochastic_processes.arrival_models import *
from mbt_gym.stochastic_processes.fill_probability_models import *
from mbt_gym.gym.ModelDynamics import LimitOrderModelDynamics
from mbt_gym.gym.helpers.plotting import *
from mbt_gym.gym.helpers.visualize_return import *
from mbt_gym.gym.helpers.generate_trajectory import generate_trajectory
from mbt_gym.gym.index_names import *
from mbt_gym.agents.SbAgent import SbAgent
from mbt_gym.agents.BaselineAgents import *

from env_utils import *

def run_single_episode(args):
    """
    Runs one evaluation episode for one seed as a worker in the multiprocessing metrics pipeline.
    """
    try:
        assert len(args) == 6, "[ERROR] Incorrect number of arguments provided"
        model_type, model_path, agent_params, env_config, hyperparams, seed = args

        # Initialize environment
        simulation_env = get_env_myModel()
    
        # Model loading logic
        if model_type in ["A2C", "SAC", "PPO"]:
            assert Path(model_path).exists(), f"[ERROR] Model path does not exist: {model_path}"

        # RL model loading
        if model_type in ["A2C", "SAC", "PPO"]:
            assert Path(model_path).exists(), f"[ERROR] Model path does not exist: {model_path}"
            model = {"A2C": A2C, "SAC": SAC, "PPO": PPO}[model_type].load(model_path)
            assert model is not None, "[ERROR] The RL model was not loaded correctly"
            agent = SbAgent(model, **agent_params)

        elif model_type == "Avellaneda-Stoikov":
            agent = AvellanedaStoikovAgent(risk_aversion=0.1, env=simulation_env)
        else:
            raise ValueError(f"[ERROR] Unsupported model type: {model_type}")

        assert agent is not None, "[ERROR] Agent not initialized correctly"

        # Seeding
        np.random.seed(seed)

        # Run trajectory
        observations, actions, rewards = generate_trajectory(simulation_env, agent, seed=seed)

        # Convert to torch tensor (on CPU to avoid multiprocessing CUDA errors)
        cum_rewards = np.cumsum(np.squeeze(rewards))
        inventory = observations[:, INVENTORY_INDEX, :].flatten()

        # Extract metrics
        terminal_reward = cum_rewards[-1]
        eps_inventory_exposure = np.mean(np.abs(inventory))
        raw_terminal_inventory = inventory[-1]
        abs_terminal_inventory = np.abs(raw_terminal_inventory)

        return (terminal_reward, eps_inventory_exposure, raw_terminal_inventory, abs_terminal_inventory)
    except Exception as e:
        output_file = Path("error.log")

        # Always overwrite the log on error
        with open(output_file, "w") as f:
            f.write(f"Error: {e}\n")
            f.write(traceback.format_exc())
