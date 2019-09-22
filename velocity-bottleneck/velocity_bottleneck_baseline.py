"""File demonstrating formation of congestion in bottleneck."""
from flow.core.params import SumoParams, EnvParams, NetParams, InitialConfig, \
    InFlows, SumoLaneChangeParams, SumoCarFollowingParams
from flow.core.params import VehicleParams
from flow.core.params import TrafficLightParams

from flow.networks.bottleneck import BottleneckNetwork
from flow.controllers import SimLaneChangeController, ContinuousRouter
from flow.envs.bottleneck import BottleneckEnv
from flow.core.experiment import Experiment
import logging
results_dir = "~/flow-lab/velocity-bottleneck/baseline_results/"

import numpy as np
import pandas as pd
SCALING = 1
DISABLE_TB = True

# If set to False, ALINEA will control the ramp meter
DISABLE_RAMP_METER = True
INFLOW = 2300


class BottleneckDensityExperiment(Experiment):
    """Experiment object for bottleneck-specific simulations.

    Extends flow.core.experiment.Experiment
    """

    def __init__(self, env):
        """Instantiate the bottleneck experiment."""
        super().__init__(env)

    def run(self, num_runs, num_steps, rl_actions=None, convert_to_csv=False):
        """See parent class."""
        info_dict = {}
        if rl_actions is None:

            def rl_actions(*_):
                return None

        rets = []
        mean_rets = []
        ret_lists = []
        vels = []
        mean_vels = []
        std_vels = []
        mean_densities = []
        mean_outflows = []
        for i in range(num_runs):
            vel = np.zeros(num_steps)
            logging.info('Iter #' + str(i))
            ret = 0
            ret_list = []
            step_outflows = []
            step_densities = []
            state = self.env.reset()
            for j in range(num_steps):
                state, reward, done, _ = self.env.step(rl_actions(state))
                vel[j] = np.mean(self.env.k.vehicle.get_speed(
                    self.env.k.vehicle.get_ids()))
                ret += reward
                ret_list.append(reward)

                env = self.env
                step_outflow = env.k.vehicle.get_outflow_rate(500)
                density = self.env.get_bottleneck_density()

                step_outflows.append(step_outflow)
                step_densities.append(density)
                if done:
                    break
            rets.append(ret)
            vels.append(vel)
            mean_densities.append(sum(step_densities[100:]) /
                                  (num_steps - 100))
            env = self.env
            outflow = env.k.vehicle.get_outflow_rate(10000)
            mean_outflows.append(outflow)
            mean_rets.append(np.mean(ret_list))
            ret_lists.append(ret_list)
            mean_vels.append(np.mean(vel))
            std_vels.append(np.std(vel))
            print('Round {0}, return: {1}'.format(i, ret))

        info_dict['returns'] = rets
        info_dict['velocities'] = vels
        info_dict['mean_returns'] = mean_rets
        info_dict['per_step_returns'] = ret_lists
        info_dict['average_outflow'] = np.mean(mean_outflows)
        info_dict['per_rollout_outflows'] = mean_outflows

        info_dict['average_rollout_density_outflow'] = np.mean(mean_densities)

        print('Average, std return: {}, {}'.format(
            np.mean(rets), np.std(rets)))
        print('Average, std speed: {}, {}'.format(
            np.mean(mean_vels), np.std(std_vels)))
        self.env.terminate()

        return info_dict, mean_outflows


def bottleneck_example(flow_rate, horizon, restart_instance=False,
                       render=None):
    """
    Perform a simulation of vehicles on a bottleneck.

    Parameters
    ----------
    flow_rate : float
        total inflow rate of vehicles into the bottleneck
    horizon : int
        time horizon
    restart_instance: bool, optional
        whether to restart the instance upon reset
    render: bool, optional
        specifies whether to use the gui during execution

    Returns
    -------
    exp: flow.core.experiment.Experiment
        A non-rl experiment demonstrating the performance of human-driven
        vehicles on a bottleneck.
    """
    if render is None:
        render = False

    sim_params = SumoParams(
        sim_step=0.5,
        render=render,
        overtake_right=False,
        restart_instance=restart_instance,
        emission_path='data')

    vehicles = VehicleParams()

    vehicles.add(
        veh_id="human",
        lane_change_controller=(SimLaneChangeController, {}),
        routing_controller=(ContinuousRouter, {}),
        car_following_params=SumoCarFollowingParams(
            speed_mode=25,
        ),
        lane_change_params=SumoLaneChangeParams(
            lane_change_mode=1621,
        ),
        num_vehicles=1)

    # print(ADDITIONAL_ENV_PARAMS)
    # {'max_accel': 3, 'max_decel': 3, 'lane_change_duration': 5, 'disable_tb': True, 'disable_ramp_metering': True}
    additional_env_params = {
        "target_velocity": 40,
        "max_accel": 1,
        "max_decel": 1,
        "lane_change_duration": 5,
        "add_rl_if_exit": False,
        "disable_tb": DISABLE_TB,
        "disable_ramp_metering": DISABLE_RAMP_METER
    }
    env_params = EnvParams(
        horizon=horizon, additional_params=additional_env_params)

    inflow = InFlows()
    inflow.add(
        veh_type="human",
        edge="1",
        vehsPerHour=flow_rate,
        departLane="random",
        departSpeed=10)

    traffic_lights = TrafficLightParams()
    if not DISABLE_TB:
        traffic_lights.add(node_id="2")
    if not DISABLE_RAMP_METER:
        traffic_lights.add(node_id="3")

    # print(ADDITIONAL_NET_PARAMS)
    # we use the default {'scaling': 1, 'speed_limit': 23}
    additional_net_params = {"scaling": SCALING, "speed_limit": 23}
    net_params = NetParams(
        inflows=inflow,
        additional_params=additional_net_params)

    initial_config = InitialConfig(
        spacing="random",
        min_gap=5,
        lanes_distribution=float("inf"),
        edges_distribution=["2", "3", "4", "5"])

    network = BottleneckNetwork(
        name="bay_bridge_toll",
        vehicles=vehicles,
        net_params=net_params,
        initial_config=initial_config,
        traffic_lights=traffic_lights)

    env = BottleneckEnv(env_params, sim_params, network)

    return BottleneckDensityExperiment(env)
    #visualize plot here [needs tuning]


if __name__ == '__main__':

    # import the experiment variable
    tested_flow_rates = 400
    each_flow_data = {"FLOW_IN": [], "ALL_FLOW_OUT": []}
    flow_in = []
    flow_out = []
    while tested_flow_rates < 2600:
        exp = bottleneck_example(tested_flow_rates, 2000)
        info, outflows = exp.run(10, 2000)
        # store info
        flow_in += [tested_flow_rates]
        flow_out += [info['average_outflow']]
        each_flow_data["FLOW_IN"].extend([tested_flow_rates] * len(outflows))
        each_flow_data["ALL_FLOW_OUT"].extend(outflows)
        # update flow rate
        tested_flow_rates += 100

    # storing the data in csv files
    mean_flow_data = {'FLOW_IN': flow_in, 'MEAN_FLOW_OUT': flow_out}
    path_for_mean_results = results_dir + "ramp_off_mean.csv"
    path_for_each_results = results_dir + "ramp_off_each.csv"
    pd.DataFrame(mean_flow_data).to_csv(path_for_mean_results, index=False)
    pd.DataFrame(each_flow_data).to_csv(path_for_each_results, index=False)


# (from paper)
# To compute this, we swept
# over inflows from 400 to 2500 in steps of 100, ran 10 runs
# for each inflow value, and stored the average outflow over
# the last 500 seconds. Fig. 6 presents the average value, 1
# std-deviation from the average,
