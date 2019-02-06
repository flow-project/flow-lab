"""Used as an example of sugiyama experiment.

This example consists of 22 IDM cars on a ring creating shockwaves.
"""
from utils import ControllerEnv
from flow.controllers import IDMController, ContinuousRouter, PISaturation
from flow.core.experiment import Experiment
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import VehicleParams
from flow.core.params import SumoCarFollowingParams
from flow.scenarios.loop import LoopScenario, ADDITIONAL_NET_PARAMS

# number of steps until a vehicle begin acting as an automated vehicles
AUTOMATED_TIME = 3000
# length of the ring during the simulation
LENGTH = 260


def pisaturation_example(render=None):
    """Perform a simulation of vehicles on a ring road.

    In this simulation, one vehicle is automated, and switches from
    human-driving to begins to automated vehicles after the time specified in
    AUTOMATED_TIME.

    Parameters
    ----------
    render : bool, optional
        specifies whether to use the gui during execution

    Returns
    -------
    exp: flow.core.experiment.Experiment
        A non-rl experiment demonstrating the performance of human-driven
        vehicles on a ring road.
    """
    sim_params = SumoParams(sim_step=0.1, render=True)

    if render is not None:
        sim_params.render = render

    vehicles = VehicleParams()
    vehicles.add(
        veh_id='automated',
        acceleration_controller=(PISaturation, {}),
        car_following_params=SumoCarFollowingParams(accel=1),
        routing_controller=(ContinuousRouter, {}),
        num_vehicles=1)
    vehicles.add(
        veh_id='human',
        acceleration_controller=(IDMController, {
            'noise': 0.2}),
        routing_controller=(ContinuousRouter, {}),
        num_vehicles=21)

    env_params = EnvParams(
        additional_params={
            'automated_time': AUTOMATED_TIME
        }
    )

    additional_net_params = ADDITIONAL_NET_PARAMS.copy()
    additional_net_params['length'] = LENGTH
    net_params = NetParams(additional_params=additional_net_params)

    initial_config = InitialConfig()

    scenario = LoopScenario(
        name='sugiyama',
        vehicles=vehicles,
        net_params=net_params,
        initial_config=initial_config)

    env = ControllerEnv(env_params, sim_params, scenario)

    return Experiment(env)


if __name__ == "__main__":
    # import the experiment variable
    exp = pisaturation_example()

    # run for a set number of rollouts / time steps
    exp.run(1, 6000)
