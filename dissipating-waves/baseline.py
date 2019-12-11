# from flow.envs import TestEnv
from flow.envs.merge import MergePOEnv, ADDITIONAL_ENV_PARAMS
from flow.controllers import RLController, IDMController
from flow.core.experiment import Experiment
from flow.core.params import SumoParams, EnvParams, InitialConfig, InFlows, NetParams
from flow.core.params import VehicleParams, SumoCarFollowingParams
from flow.scenarios import MergeScenario
from flow.scenarios.merge import ADDITIONAL_NET_PARAMS

RL_PENETRATION = 0.1
FLOW_RATE = 2000


def dissipating_waves(render=None):

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
        num_vehicles=5)
    vehicles.add(
        veh_id="rl",
        acceleration_controller=(RLController, {}),
        car_following_params=SumoCarFollowingParams(
            speed_mode="obey_safe_speed",
        ),
        num_vehicles=0)

    # Vehicles are introduced from both sides of merge, with RL vehicles entering
    # from the highway portion as well
    inflow = InFlows()
    inflow.add(
        veh_type="human",
        edge="inflow_highway",
        vehs_per_hour=(1 - RL_PENETRATION) * FLOW_RATE,
        depart_lane="free",
        depart_speed=10)
    inflow.add(
        veh_type="rl",
        edge="inflow_highway",
        vehs_per_hour=RL_PENETRATION * FLOW_RATE,
        depart_lane="free",
        depart_speed=10)
    inflow.add(
        veh_type="human",
        edge="inflow_merge",
        vehs_per_hour=100,
        depart_lane="free",
        depart_speed=7.5)

    # Set parameters for the network
    additional_net_params = ADDITIONAL_NET_PARAMS.copy()
    additional_net_params["pre_merge_length"] = 500
    additional_net_params["post_merge_length"] = 200
    additional_net_params["merge_lanes"] = 1
    additional_net_params["highway_lanes"] = 1
    net_params = NetParams(inflows=inflow,
                           additional_params=additional_net_params)

    # Setup the scenario
    initial_config = InitialConfig()
    scenario = MergeScenario(
        name='testing',
        vehicles=vehicles,
        net_params=net_params,
        initial_config=initial_config)

    # Setup the environment
    env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)
    env = MergePOEnv(env_params, sim_params, scenario)
    return Experiment(env)

if __name__=="__main__":
    exp = dissipating_waves(render=True)
    exp.run(1, 6000)
