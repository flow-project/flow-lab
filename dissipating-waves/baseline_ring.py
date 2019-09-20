# from flow.envs import TestEnv
from utils import make_create_env, PerturbingRingEnv, PERTURB_ENV_PARAMS
from flow.controllers import RLController, IDMController, ContinuousRouter
from flow.core.experiment import Experiment
from flow.core.params import SumoParams, EnvParams, InitialConfig, InFlows, NetParams
from flow.core.params import VehicleParams, SumoCarFollowingParams
from flow.networks import RingNetwork
from flow.networks.ring import ADDITIONAL_NET_PARAMS


def ring_perturbation(render=None):

    sim_params = SumoParams(sim_step=0.2, render=True, restart_instance=True)

    if render is not None:
        sim_params.render = render

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
                accel=1.5, # vehicle does not inherit from env_params
                decel=1.5,
            ),
            num_vehicles=1)


    # Set parameters for the network
    additional_net_params = ADDITIONAL_NET_PARAMS.copy()
    additional_net_params["length"] = 1400
    additional_net_params["speed_limit"] = 20
    net_params = NetParams(additional_params=additional_net_params)

    # Setup the network
    initial_config = InitialConfig()
    network = RingNetwork(
        name='testing',
        vehicles=vehicles,
        net_params=net_params,
        initial_config=initial_config)

    # Setup the environment
    PERTURB_ENV_PARAMS["merge_flow_rate"] = 100
    env_params = EnvParams(additional_params=PERTURB_ENV_PARAMS)
    env = PerturbingRingEnv(env_params, sim_params, network)
    return Experiment(env)

if __name__=="__main__":
    exp = ring_perturbation(render=True)
    exp.run(1, 1000)
