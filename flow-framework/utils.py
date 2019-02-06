from flow.envs import TestEnv
from flow.controllers import IDMController, FollowerStopper, PISaturation
from flow.core.params import SumoCarFollowingParams


class ControllerEnv(TestEnv):
    """Environment that highlights specific controllers.

    This is used to highlight *PI with Saturation* and *FollowerStopper*
    controllers in red, while all other cars are in white.

    Extends flow.envs.TestEnv
    """

    def __init__(self, env_params, sim_params, scenario, simulator='traci'):
        super().__init__(env_params, sim_params, scenario, simulator)
        self.automated_time = env_params.additional_params['automated_time']
        self.controllers = dict()

    def additional_command(self):
        self.sim_params.render = False
        for veh_id in self.k.vehicle.get_ids():
            # check if vehicle is an automated vehicle
            if type(self.k.vehicle.get_acc_controller(veh_id)) \
                    in [FollowerStopper, PISaturation]:
                # set the color as red
                self.k.vehicle.set_color(veh_id, color=(255, 0, 0))
                # if below automated time, use IDM to control vehicle dynamics
                if self.time_counter < self.automated_time:
                    accel = self.controllers[veh_id].get_accel(self)
                    self.k.vehicle.apply_acceleration([veh_id], [accel])

    def reset(self):
        obs = super().reset()
        self.controllers.clear()
        for veh_id in self.k.vehicle.get_ids():
            if type(self.k.vehicle.get_acc_controller(veh_id)) \
                    in [FollowerStopper, PISaturation]:
                self.controllers[veh_id] = IDMController(
                    veh_id=veh_id,
                    car_following_params=SumoCarFollowingParams())
        return obs
