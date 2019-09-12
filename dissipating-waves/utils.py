from copy import deepcopy
import gym
from gym.envs.registration import register
import flow.envs
from flow.envs import MergePOEnv
from flow.core.params import InitialConfig
from flow.core.params import TrafficLightParams


# Use while #718 in flow is not yet resolved
def make_create_env(params, version=0, render=None):
    exp_tag = params["exp_tag"]

    env_name = params["env_name"] + '-v{}'.format(version)

    module = __import__("flow.scenarios", fromlist=[params["scenario"]])
    scenario_class = getattr(module, params["scenario"])

    env_params = params['env']
    net_params = params['net']
    initial_config = params.get('initial', InitialConfig())
    traffic_lights = params.get("tls", TrafficLightParams())

    def create_env(*_):
        sim_params = deepcopy(params['sim'])
        vehicles = deepcopy(params['veh'])

        scenario = scenario_class(
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

        if params['env_name'] in single_agent_envs:
            env_loc = 'flow.envs'
        else:
            env_loc = 'flow.envs.multiagent'

        try:
            register(
                id=env_name,
                entry_point=env_loc + ':{}'.format(params["env_name"]),
                kwargs={
                    "env_params": env_params,
                    "sim_params": sim_params,
                    "scenario": scenario,
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

    return max(max_cost - cost, 0)


class unscaledMergePOEnv(MergePOEnv):

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
