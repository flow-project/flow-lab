import json

import ray
try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class
from ray.tune import run_experiments
from ray.tune.registry import register_env

# from flow.utils.registry import make_create_env
# from flow.utils.rllib import FlowParamsEncoder
from utils import PerturbingRingEnv, make_create_env, FlowParamsEncoder, PERTURB_ENV_PARAMS
from flow.controllers import RLController, IDMController, ContinuousRouter
from flow.core.experiment import Experiment
from flow.core.params import SumoParams, EnvParams, InitialConfig, InFlows, NetParams
from flow.core.params import VehicleParams, SumoCarFollowingParams
from flow.networks.ring import ADDITIONAL_NET_PARAMS

HORIZON = 18000
N_ROLLOUTS = 20
N_CPUS = 16
ACCEL = 1.5


# Setup vehicle types
vehicles = VehicleParams()
for i in range(5):
    vehicles.add(
        veh_id="human{}".format(i),
        acceleration_controller=(IDMController, {
            "noise": 0.2
        }),
        routing_controller=(ContinuousRouter, {}),
        car_following_params=SumoCarFollowingParams(
            speed_mode="obey_safe_speed",
            min_gap=0.5, 
        ),
        num_vehicles=9)
    vehicles.add(
        veh_id="rl{}".format(i),
        acceleration_controller=(RLController, {}),
        routing_controller=(ContinuousRouter, {}),
        car_following_params=SumoCarFollowingParams(
            speed_mode="obey_safe_speed",
            accel=ACCEL, # vehicle does not inherit from env_params
            decel=ACCEL,
        ),
        num_vehicles=1)

# Set parameters for the network
additional_net_params = ADDITIONAL_NET_PARAMS.copy()
additional_net_params["length"] = 1400
additional_net_params["speed_limit"] = 20


flow_params =  dict(
        # name of the experiment
        exp_tag="perturbing_ring",

        # name of the flow environment the experiment is running on
        env_name=PerturbingRingEnv,

        # name of the network class the experiment is running on
        network="RingNetwork",

        # simulator that is used by the experiment
        simulator='traci',

        # sumo-related parameters (see flow.core.params.SumoParams)
        sim=SumoParams(
            sim_step=0.2,
            render=False,
            restart_instance=True,
        ),

        # environment related parameters (see flow.core.params.EnvParams)
        env=EnvParams(
            horizon=HORIZON,
            sims_per_step=5,
            warmup_steps=0,
            additional_params=PERTURB_ENV_PARAMS,
        ),

        # network-related parameters (see flow.core.params.NetParams and the
        # network's documentation or ADDITIONAL_NET_PARAMS component)
        net=NetParams(
            additional_params=additional_net_params,
        ),

        # vehicles to be placed in the network at the start of a rollout (see
        # flow.core.params.VehicleParams)
        veh=vehicles,

        # parameters specifying the positioning of vehicles upon initialization/
        # reset (see flow.core.params.InitialConfig)
        initial=InitialConfig(),
    )


def setup_exps():
    """Return the relevant components of an RLlib experiment.

    Returns
    -------
    str
        name of the training algorithm
    str
        name of the gym environment to be trained
    dict
        training configuration parameters
    """
    alg_run = "PPO"

    agent_cls = get_agent_class(alg_run)
    config = agent_cls._default_config.copy()
    config["num_workers"] = N_CPUS
    config["train_batch_size"] = HORIZON * N_ROLLOUTS
    config["gamma"] = 0.999  # discount rate
    config["model"].update({"fcnet_hiddens": [32,32,32]})
    config["use_gae"] = True
    config["lambda"] = 0.97
    config["kl_target"] = 0.02
    config["num_sgd_iter"] = 10
    # config['clip_actions'] = False  # FIXME(ev) temporary ray bug
    config["horizon"] = HORIZON

    # save the flow params for replay
    flow_json = json.dumps(
        flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4)
    config['env_config']['flow_params'] = flow_json
    config['env_config']['run'] = alg_run

    create_env, gym_name = make_create_env(params=flow_params, version=0)

    # Register as rllib env
    register_env(gym_name, create_env)
    return alg_run, gym_name, config


if __name__ == "__main__":
    import functools
    import os
    ray.init(num_cpus=N_CPUS + 1, redirect_output=False)
    alg_run, gym_name, config = setup_exps()
    trials = run_experiments({
        flow_params["exp_tag"]: {
            "run": alg_run,
            "env": gym_name,
            "config": {
                **config
            },
            "checkpoint_freq": 20,
            "checkpoint_at_end": True,
            "max_failures": 999,
            "stop": {
                "training_iteration": 200,
            },
            "local_dir": os.path.abspath("./ray_results")
        }
    })
