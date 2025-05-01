from typing import Dict

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np
from hydrax.files import get_root_path
from hydrax.task_base import Task


class Particle(Task):
    """A velocity-controlled planar point mass chases a target position."""

    def __init__(
        self, planning_horizon: int = 20, sim_steps_per_control_step: int = 5
    ):
        """Load the MuJoCo model and set task parameters."""
        mj_model = mujoco.MjModel.from_xml_path(
            (get_root_path() / "hydrax" / "models" / "particle" / "scene.xml").as_posix()
        )

        super().__init__(
            mj_model,
            planning_horizon=planning_horizon,
            sim_steps_per_control_step=sim_steps_per_control_step,
            trace_sites=["pointmass"],
        )
        self.wall_pos = jnp.array([
            mj_model.geom("wall_ix").pos[:2],
            mj_model.geom("wall_iy").pos[:2],
            mj_model.geom("wall_neg_iy").pos[:2],
        ])
        self.wall_size = jnp.array([
            mj_model.geom("wall_ix").size[:2],
            mj_model.geom("wall_iy").size[:2],
            mj_model.geom("wall_neg_iy").size[:2],
        ])
        
        self.pointmass_id = mj_model.site("pointmass").id

    def reset(self, seed: int = 0) -> None:
        np.random.seed(seed)
        self.task_success = False
        mj_data = mujoco.MjData(self.mj_model)
        base_pos = np.array([-0.2, 0.0])
        base_pos += np.random.randn(2) * 0.02
        mj_data.qpos[:2] = base_pos
        return self.mj_model, mj_data

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ) encourages target tracking."""
        # wall SDF cost
        wall_dist = jnp.abs(state.site_xpos[self.pointmass_id][None, :2] - self.wall_pos) - self.wall_size
        outside_dist = jnp.maximum(wall_dist, 1e-12)
        inside_dist = jnp.minimum(jnp.max(wall_dist, axis=-1), 0.0)
        dist = (jnp.linalg.norm(outside_dist, axis=-1) + inside_dist).min(axis=-1)
        state_cost = 5. * jnp.exp(-50. * dist)
        state_cost += self.terminal_cost(state)
        control_cost = jnp.sum(jnp.square(control))
        return state_cost + 0.1 * control_cost

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T)."""
        position_cost = jnp.sum(
            jnp.square(state.site_xpos[self.pointmass_id] - state.mocap_pos[0])
        )
        velocity_cost = jnp.sum(jnp.square(state.qvel))
        return 5.0 * position_cost + 0.1 * velocity_cost
    
    def success(self, state):
        position_cost = jnp.sum(
            jnp.square(state.site_xpos[self.pointmass_id] - state.mocap_pos[0])
        )
        return jnp.sqrt(position_cost) < self.success_threshold

    def domain_randomize_model(self, rng: jax.Array) -> Dict[str, jax.Array]:
        """Randomly perturb the actuator gains."""
        multiplier = jax.random.uniform(
            rng, self.model.actuator_gainprm[:, 0].shape, minval=0.9, maxval=1.1
        )
        new_gains = self.model.actuator_gainprm[:, 0] * multiplier
        new_gains = self.model.actuator_gainprm.at[:, 0].set(new_gains)
        return {"actuator_gainprm": new_gains}

    def domain_randomize_data(
        self, data: mjx.Data, rng: jax.Array
    ) -> Dict[str, jax.Array]:
        """Randomly shift the measured particle position."""
        shift = jax.random.uniform(rng, (2,), minval=-0.01, maxval=0.01)
        return {"qpos": data.qpos + shift}
