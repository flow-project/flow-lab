from flow.networks.bottleneck import BottleneckNetwork, ADDITIONAL_NET_PARAMS
from flow.core.params import VehicleParams, InFlows, SumoCarFollowingParams, SumoLaneChangeParams
from flow.controllers.car_following_models import IDMController
from flow.controllers import SimLaneChangeController, ContinuousRouter, RLController
from flow.core.params import NetParams
from flow.core.params import InitialConfig
from flow.core.params import TrafficLightParams
from flow.core.params import SumoParams
from flow.core.params import EnvParams
from flow.envs.bottleneck import BottleneckEnv
from flow.envs.bottleneck import ADDITIONAL_ENV_PARAMS
from flow.core.experiment import Experiment

#wrap in function
"""This network, as well as all other networks in Flow, is parametrized by the following arguments:
name _/
vehicles_/
net_params_/
initial_config_/
traffic_lights_/"""

name = "velocity_bottleneck_baseline"
vehicles = VehicleParams()

#specify your controllers
vehicles.add("human",
             acceleration_controller= (IDMController, {}),
             car_following_params=SumoCarFollowingParams(speed_mode="obey_safe_speed"),
             # lane_change_controller= (SimLaneChangeController, {}),
             num_vehicles = 20)
#test adding an rl vehicle
vehicles.add("rl",
             acceleration_controller= (RLController, {}),
             car_following_params=SumoCarFollowingParams(speed_mode="obey_safe_speed"),
             # lane_change_controller= (SimLaneChangeController, {}),
             num_vehicles = 3)

inflow = InFlows()
inflow.add(veh_type="human",
           edge="1",
           depart_lane="random",
           vehs_per_hour=2300)
inflow.add(veh_type="rl",
           edge="1",
           depart_lane="random",
           vehs_per_hour=400)
# inflow.add(
#     veh_type="human",
#     edge="1",
#     vehsPerHour=2300,
#     departLane="random",
#     departSpeed=10)
# print(ADDITIONAL_NET_PARAMS)
#we use the default {'scaling': 1, 'speed_limit': 23}

#specify network parameters
net_params = NetParams(inflows = inflow, additional_params=ADDITIONAL_NET_PARAMS)

#what's happening here? check
initial_config = InitialConfig(spacing="random",
                               shuffle=True,
                        min_gap=5,
                        # lanes_distribution=float("inf"),
                        edges_distribution=["2", "3", "4", "5"]) #need to figure these parameters out

traffic_lights = TrafficLightParams()

#Create an nvironment
# EnvParams
# SumoParams _/
# Network_/
sumo_params = SumoParams(sim_step=0.1, render=True, overtake_right=False, restart_instance=False) #extra parameters
env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)

#scenario/network takes inputs object 1
network = BottleneckNetwork(name=name,
                      vehicles=vehicles,
                      net_params=net_params,
                      initial_config=initial_config,
                      traffic_lights=traffic_lights)

#object 2
env = BottleneckEnv(env_params, sumo_params, network)

#object 3
exp = Experiment(env)

#run
exp.run(1, 10000)
