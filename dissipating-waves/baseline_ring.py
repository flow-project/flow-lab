# from flow.envs import TestEnv
from utils import make_create_env, PerturbingRingEnv, PERTURB_ENV_PARAMS
from flow.controllers import RLController, IDMController, ContinuousRouter
from flow.core.experiment import Experiment
from flow.core.params import SumoParams, EnvParams, InitialConfig, InFlows, NetParams
from flow.core.params import VehicleParams, SumoCarFollowingParams
from flow.networks import RingNetwork
from flow.networks.ring import ADDITIONAL_NET_PARAMS


def ring_perturbation(render=None):

    sim_params = SumoParams(sim_step=0.1, render=True, restart_instance=True)

    if render is not None:
        sim_params.render = render

    # Setup vehicle types
    vehicles = VehicleParams()
    vehicles.add(
        veh_id="human",
        acceleration_controller=(IDMController, {
            "noise": 0.2
        }),
        car_following_params=SumoCarFollowingParams(
            speed_mode="obey_safe_speed",
        ),
        routing_controller=(ContinuousRouter, {}),
        num_vehicles=45)
    # For debugging only
    vehicles.add(
        veh_id="rl",
        acceleration_controller=(RLController, {}),
        car_following_params=SumoCarFollowingParams(
            speed_mode="obey_safe_speed",
        ),
        routing_controller=(ContinuousRouter, {}),
        num_vehicles=5)


    # Set parameters for the network
    additional_net_params = ADDITIONAL_NET_PARAMS.copy()
    additional_net_params["length"] = 1400
    additional_net_params["speed_limit"] = 30
    net_params = NetParams(
                           additional_params=additional_net_params)

    # Setup the network
    initial_config = InitialConfig(shuffle=True)
    network = RingNetwork(
        name='testing',
        vehicles=vehicles,
        net_params=net_params,
        initial_config=initial_config)

    # Setup the environment
    env_params = EnvParams(additional_params=PERTURB_ENV_PARAMS)
    env = PerturbingRingEnv(env_params, sim_params, network)
    return Experiment(env)

if __name__=="__main__":
    exp = ring_perturbation(render=True)
    exp.run(1, 1000)
