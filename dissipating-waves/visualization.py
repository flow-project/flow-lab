import argparse
from datetime import datetime
import gym
import numpy as np
import os
import sys
import time

import ray
try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class
from ray.tune.registry import register_env

from flow.core.util import emission_to_csv
from flow.utils.registry import make_create_env
from flow.utils.rllib import get_flow_params
from flow.utils.rllib import get_rllib_config
from flow.utils.rllib import get_rllib_pkl


def reload_checkpoint(result_dir, checkpoint_num, gen_emission=False, version=0):
    config = get_rllib_config(result_dir)

    # Run on only one cpu for rendering purposes
    config['num_workers'] = 0

    flow_params = get_flow_params(config)

    # hack for old pkl files
    # TODO(ev) remove eventually
    sim_params = flow_params['sim']
    setattr(sim_params, 'num_clients', 1)

    # Determine agent and checkpoint
    config_run = config['env_config'].get("run", None)
    agent_cls = get_agent_class(config_run)
        
    sim_params.restart_instance = True
    dir_path = os.path.dirname(os.path.realpath(__file__))
    emission_path = '{0}/emission/'.format(dir_path)
    sim_params.emission_path = emission_path if gen_emission else None

    # pick your rendering mode
    sim_params.render = False
    create_env, env_name = make_create_env(params=flow_params, version=version)
    register_env(env_name, create_env)

    env_params = flow_params['env']
    env_params.restart_instance = False

    # create the agent that will be used to compute the actions
    agent = agent_cls(env=env_name, config=config)
    checkpoint = result_dir + '/checkpoint_{}'.format(checkpoint_num)
    checkpoint = checkpoint + '/checkpoint-{}'.format(checkpoint_num)
    agent.restore(checkpoint)

    env = agent.local_evaluator.env

    env.restart_simulation(
        sim_params=sim_params, render=sim_params.render)

    return env, env_params, agent

def replay(env, env_params, agent):
    # Replay simulations
    state = env.reset()
    for _ in range(env_params.horizon):
        vehicles = env.unwrapped.k.vehicle
        action = agent.compute_action(state)
        state, reward, done, _ = env.step(action)
        if done:
            break

    outflow = vehicles.get_outflow_rate(500)
    inflow = vehicles.get_inflow_rate(500)
    throughput_efficiency = outflow/inflow if inflow>1e-5 else 0
        
    # terminate the environment
    env.unwrapped.terminate()

def get_emission_csv(result_dir):
    time.sleep(0.1)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    emission_filename = '{0}-emission.xml'.format(env.network.name)
    emission_path = \
        '{0}/emission/{1}'.format(dir_path, emission_filename)

    # convert the emission file into a csv file
    emission_to_csv(emission_path)

    # delete the .xml version of the emission file
    os.remove(emission_path)

    # # sometimes it takes a while to convert
    # while not os.path.exists(emission_path[:-3]+"csv"):
    #     time.sleep(1)
    os.rename(emission_path[:-3]+"csv", "emission/{}.csv".format(result_dir.split("/")[-1]))

if __name__=="__main__":
    ray.init(num_cpus=1)

    experiment_dir = "ray_results/dissipating_waves"
    result_dirs = os.listdir(experiment_dir)
    for i, result_dir in enumerate(result_dirs):
        if result_dir[0]=='.': continue
        result_dir = "{}/{}".format(experiment_dir, result_dir)
        print("Processing {}".format(result_dir))

        # version is important so that we are sure we are using different environments
        # as we iterate over the 3 experiments.
        env, env_params, agent = reload_checkpoint(result_dir, 20, True, version=i)
        replay(env, env_params, agent)
        get_emission_csv(result_dir)
