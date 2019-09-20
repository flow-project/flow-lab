from copy import deepcopy
import gym
from gym.envs.registration import register
import flow.envs
from flow.envs import MergePOEnv, Env
from flow.core.params import InitialConfig, VehicleParams
from flow.core.params import TrafficLightParams
import numpy as np
import inspect
import json

## need to override the encoder
class FlowParamsEncoder(json.JSONEncoder):
    """modified from flow.utils.rllib"""
    def default(self, obj):
        """See parent class.

        Extended to support the VehicleParams object in flow/core/params.py.
        """
        allowed_types = [dict, list, tuple, str, int, float, bool, type(None)]

        if obj not in allowed_types:
            if isinstance(obj, VehicleParams):
                res = deepcopy(obj.initial)
                for res_i in res:
                    res_i["acceleration_controller"] = \
                        (res_i["acceleration_controller"][0].__name__,
                         res_i["acceleration_controller"][1])
                    res_i["lane_change_controller"] = \
                        (res_i["lane_change_controller"][0].__name__,
                         res_i["lane_change_controller"][1])
                    if res_i["routing_controller"] is not None:
                        res_i["routing_controller"] = \
                            (res_i["routing_controller"][0].__name__,
                             res_i["routing_controller"][1])
                return res
            if inspect.isclass(obj):
                if issubclass(obj, Env):
                    return "{}.{}".format(obj.__module__, obj.__name__)
            if hasattr(obj, '__name__'):
                return obj.__name__
            else:
                return obj.__dict__

        return json.JSONEncoder.default(self, obj)


# Use while #718 in flow is not yet resolved
def make_create_env(params, version=0, render=None):
    exp_tag = params["exp_tag"]

    try:
        env_name = params["env_name"].__name__ + '-v{}'.format(version)
    except AttributeError:
        if "." in params['env_name']:
            env_name = params['env_name'].split(".")[-1] + '-v{}'.format(version)
        env_name = params["env_name"] + '-v{}'.format(version)

    module = __import__("flow.networks", fromlist=[params["network"]])
    network_class = getattr(module, params["network"])

    env_params = params['env']
    net_params = params['net']
    initial_config = params.get('initial', InitialConfig())
    traffic_lights = params.get("tls", TrafficLightParams())

    def create_env(*_):
        sim_params = deepcopy(params['sim'])
        vehicles = deepcopy(params['veh'])

        network = network_class(
            name=exp_tag,
            vehicles=vehicles,
            net_params=net_params,
            initial_config=initial_config,
            traffic_lights=traffic_lights,
        )

        # accept new render type if not set to None
        sim_params.render = render or sim_params.render

        # check if the environment is a single or multiagent environment, and
        # get the right address accordingly
        single_agent_envs = [env for env in dir(flow.envs)
                             if not env.startswith('__')]

        try:
            entry_point = params["env_name"].__module__ + ':' + params["env_name"].__name__
        except AttributeError:
            # When loading a trained policy, it will be serielized back to a string
            if "." in params['env_name']:
                env_loc = ".".join(params['env_name'].split(".")[:-1])
                entry_point = env_loc + ':{}'.format(params['env_name'].split(".")[-1])
            else:
                if params['env_name'] in single_agent_envs:
                    env_loc = 'flow.envs'
                else:
                    env_loc = 'flow.envs.multiagent'
                entry_point = env_loc + ':{}'.format(params["env_name"])

        try:
            register(
                id=env_name,
                entry_point=entry_point,
                kwargs={
                    "env_params": env_params,
                    "sim_params": sim_params,
                    "network": network,
                    "simulator": params['simulator']
                })
        except Exception:
            pass
        return gym.envs.make(env_name)

    return create_env, env_name


def desired_velocity(env, fail=False, edge_list=None):
    if edge_list is None:
        veh_ids = env.k.vehicle.get_ids()
    else:
        veh_ids = env.k.vehicle.get_ids_by_edge(edge_list)

    vel = np.array(env.k.vehicle.get_speed(veh_ids))
    num_vehicles = len(veh_ids)

    if any(vel < -100) or fail or num_vehicles == 0:
        return 0.

    target_vel = env.env_params.additional_params['target_velocity']
    max_cost = np.array([target_vel] * num_vehicles)
    max_cost = np.linalg.norm(max_cost)

    cost = vel - target_vel
    cost = np.linalg.norm(cost)

    #Removed rescaling -- flow.core.rewards.desired_velocity
    return max(max_cost - cost, 0)


class unscaledMergePOEnv(MergePOEnv):
    """modified from flow.envs.merge """

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        if self.env_params.evaluate:
            return np.mean(self.k.vehicle.get_speed(self.k.vehicle.get_ids()))
        else:
            # return a reward of 0 if a collision occurred
            if kwargs["fail"]:
                return 0

            # reward high system-level velocities

            cost1 = desired_velocity(self, fail=kwargs["fail"])

            # penalize small time headways
            cost2 = 0
            t_min = 1  # smallest acceptable time headway
            for rl_id in self.rl_veh:
                lead_id = self.k.vehicle.get_leader(rl_id)
                if lead_id not in ["", None] \
                        and self.k.vehicle.get_speed(rl_id) > 0:
                    t_headway = max(
                        self.k.vehicle.get_headway(rl_id) /
                        self.k.vehicle.get_speed(rl_id), 0)
                    cost2 += min((t_headway - t_min), 0)

            # weights for cost1, cost2, and cost3, respectively
            eta1, eta2 = 1.00, 0.10

            return max(eta1 * cost1 + eta2 * cost2, 0)


PERTURB_ENV_PARAMS = {
    # maximum acceleration for autonomous vehicles, in m/s^2
    "max_accel": 1.5,
    # maximum deceleration for autonomous vehicles, in m/s^2
    "max_decel": 1.5,
    # desired velocity for all vehicles in the network, in m/s
    "target_velocity": 20,
    # maximum number of controllable vehicles in the network
    "num_rl": 5,
    "merge_flow_rate": 100 # veh/hour
}

class PerturbingRingEnv(unscaledMergePOEnv):
    """Modified version of MergePOEnv that perturbs vehicles."""
    def __init__(self, env_params, sim_params, network, simulator='traci'):
        self.counter = 0
        self.num_rl_vehicles = network.vehicles.num_rl_vehicles
        self.num_total_vehicles = network.vehicles.num_vehicles
        super().__init__(env_params, sim_params, network, simulator)

    def _apply_rl_actions(self, rl_actions):
        for i, rl_id in enumerate(self.rl_veh):
            # ignore rl vehicles outside controllable region
            if self.k.vehicle.get_edge(rl_id) not in ["bottom", "left"]:
                continue
            self.k.vehicle.apply_acceleration(rl_id, rl_actions[i])

    def get_state(self, rl_id=None, **kwargs):
        self.leader = []
        self.follower = []

        # normalizing constants
        max_speed = self.k.network.max_speed()
        max_length = self.k.network.length()

        observation = [0 for _ in range(5 * self.num_rl)]
        for i, rl_id in enumerate(self.rl_veh):
            this_speed = self.k.vehicle.get_speed(rl_id)
            lead_id = self.k.vehicle.get_leader(rl_id)
            follower = self.k.vehicle.get_follower(rl_id)

            if self.k.vehicle.get_edge(lead_id) not in ["bottom", "left"]:
                # in case leader is not controllable region
                lead_speed = max_speed
                lead_head = max_length
            else:
                self.leader.append(lead_id)
                lead_speed = self.k.vehicle.get_speed(lead_id)
                lead_head = self.k.vehicle.get_x_by_id(lead_id) \
                    - self.k.vehicle.get_x_by_id(rl_id) \
                    - self.k.vehicle.get_length(rl_id)

            if self.k.vehicle.get_edge(follower) not in ["bottom", "left"]:
                # in case follower is not controllable region
                follow_speed = 0
                follow_head = max_length
            else:
                self.follower.append(follower)
                follow_speed = self.k.vehicle.get_speed(follower)
                follow_head = self.k.vehicle.get_headway(follower)

            observation[5 * i + 0] = this_speed / max_speed
            observation[5 * i + 1] = (lead_speed - this_speed) / max_speed
            observation[5 * i + 2] = lead_head / max_length
            observation[5 * i + 3] = (this_speed - follow_speed) / max_speed
            observation[5 * i + 4] = follow_head / max_length

        return observation

    def additional_command(self):
        # Do the old stuff first
        super().additional_command()

        # Do additional perturbation task
        self.counter += 1
        vph = self.env_params.additional_params['merge_flow_rate']
        period = 1 / ((vph/3600)*self.sim_params.sim_step)
        if self.counter>=period:
            self.perturb()
            self.counter = 0


    def perturb(self):
        # get vehicles in target edge (bottom?)
        in_edge = [i for i in self.k.vehicle.get_ids() if self.k.vehicle.get_edge(i)=="bottom"]

        if len(in_edge)==0: # nothing to perturb
            pass
        else:
            pos = self.k.vehicle.get_position(in_edge)
            target_id = in_edge[np.argmin(np.abs(pos))]
            self.k.vehicle.apply_acceleration(target_id, -100)