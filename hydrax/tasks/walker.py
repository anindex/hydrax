import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np
from hydrax.files import get_root_path
from hydrax.task_base import Task


class Walker(Task):
    """A planar biped tasked with walking forward."""

    def __init__(
        self, planning_horizon: int = 4, sim_steps_per_control_step: int = 15
    ):
        """Load the MuJoCo model and set task parameters."""
        mj_model = mujoco.MjModel.from_xml_path(
            (get_root_path() / "hydrax" / "models" / "walker" / "scene.xml").as_posix()
        )

        super().__init__(
            mj_model,
            planning_horizon=planning_horizon,
            sim_steps_per_control_step=sim_steps_per_control_step,
            trace_sites=["torso_site"],
        )

        # Get sensor ids
        self.torso_position_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "torso_position"
        )
        self.torso_velocity_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "torso_subtreelinvel"
        )
        self.torso_zaxis_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "torso_zaxis"
        )

        # Set the target velocity (m/s) and height
        # TODO: make these parameters
        self.target_velocity = 1.5
        self.target_height = 1.2
    
    def reset(self, seed: int = 0) -> None:
        np.random.seed(seed)        
        mj_model = self.mj_model
        mj_model.opt.timestep = 0.005
        mj_model.opt.iterations = 50
        mj_data = mujoco.MjData(mj_model)
        return mj_model, mj_data

    def _get_torso_height(self, state: mjx.Data) -> jax.Array:
        """Get the height of the torso above the ground."""
        sensor_adr = self.model.sensor_adr[self.torso_position_sensor]
        return state.sensordata[sensor_adr + 2]  # px, py, pz

    def _get_torso_velocity(self, state: mjx.Data) -> jax.Array:
        """Get the horizontal velocity of the torso."""
        sensor_adr = self.model.sensor_adr[self.torso_velocity_sensor]
        return state.sensordata[sensor_adr]

    def _get_torso_deviation_from_upright(self, state: mjx.Data) -> jax.Array:
        """Get the deviation of the torso from the upright position."""
        sensor_adr = self.model.sensor_adr[self.torso_zaxis_sensor]
        return state.sensordata[sensor_adr + 2] - 1.0

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ)."""
        state_cost = self.terminal_cost(state)
        control_cost = jnp.sum(jnp.square(control))
        return state_cost + 0.1 * control_cost

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T)."""
        height_cost = jnp.square(
            self._get_torso_height(state) - self.target_height
        )
        orientation_cost = jnp.square(
            self._get_torso_deviation_from_upright(state)
        )
        velocity_cost = jnp.square(
            self._get_torso_velocity(state) - self.target_velocity
        )
        return 10.0 * height_cost + 3.0 * orientation_cost + 1.0 * velocity_cost
