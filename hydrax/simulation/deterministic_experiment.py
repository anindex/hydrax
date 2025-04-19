import time
import csv
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx
from typing import Optional, List
from hydrax.alg_base import SamplingBasedController
from tqdm import tqdm
import os
from chrono import Timer


def run_headless_simulation(
    controller: SamplingBasedController,
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    frequency: float,
    seeds: List[int],
    delay_ctrl_start: int = 0,
    max_step: int = 10000,
    log_file_prefix: Optional[str] = None,
    save_path: Optional[str] = '.',
) -> None:
    """Run deterministic headless MuJoCo simulations with multiple seeds.

    Args:
        controller: The MPC controller instance.
        mj_model: MuJoCo simulation model.
        mj_data: MuJoCo simulation data object.
        frequency: Control frequency (Hz).
        seeds: List of seeds for multiple experiments.
        delay_ctrl_start: Steps to delay the controller's action.
        max_step: Maximum number of simulation steps.
        log_file_prefix: Prefix path to log simulation data; seed will be appended to filename.
    """
    for seed in seeds:
        print(f"\nRunning experiment with seed: {seed}")
        logs = []
        np.random.seed(seed)

        try:
            mujoco.mj_resetData(mj_model, mj_data)
            replan_period = 1.0 / frequency
            sim_steps_per_replan = int(replan_period / mj_model.opt.timestep)
            sim_steps_per_replan = max(sim_steps_per_replan, 1)

            mjx_data = mjx.put_data(mj_model, mj_data)
            mjx_data = mjx_data.replace(
                mocap_pos=mj_data.mocap_pos, mocap_quat=mj_data.mocap_quat
            )
            policy_params = controller.init_params()
            jit_optimize = jax.jit(controller.optimize, donate_argnums=(1,))

            # Controller warm-up
            policy_params, rollouts = jit_optimize(mjx_data, policy_params)

            step = 0
            for step in tqdm(range(max_step)):
                step += 1
                
                mjx_data = mjx_data.replace(
                    qpos=jnp.array(mj_data.qpos),
                    qvel=jnp.array(mj_data.qvel),
                    mocap_pos=jnp.array(mj_data.mocap_pos),
                    mocap_quat=jnp.array(mj_data.mocap_quat),
                    time=mj_data.time,
                )

                with Timer() as timer:
                    policy_params, rollouts = jit_optimize(mjx_data, policy_params)
                plan_time = timer.elapsed

                for i in range(sim_steps_per_replan):
                    t = i * mj_model.opt.timestep
                    u = controller.get_action(policy_params, t)

                    if delay_ctrl_start > 0:
                        delay_ctrl_start -= 1
                    else:
                        mj_data.ctrl[:] = np.array(u)

                    mujoco.mj_step(mj_model, mj_data)

                    if np.isnan(u).any():
                        print("NaN detected in control input; stopping current experiment.")
                        break

                logs.append({
                    "step": step,
                    "sim_time": float(mjx_data.time),
                    "plan_time": plan_time,
                    "qpos": np.array(mjx_data.qpos).tolist(),
                    "qvel": np.array(mjx_data.qvel).tolist(),
                    "control": np.array(u).tolist(),
                    "running_cost": jnp.sum(rollouts.costs, axis=1).tolist(),
                })

                if np.isnan(u).any():
                    break

        except Exception as e:
            print(f"Experiment with seed {seed} encountered an error: {e}")

        finally:
            if log_file_prefix:
                log_file = os.path.join(save_path, f"{log_file_prefix}_seed_{seed}.csv")
                with open(log_file, "w", newline="") as csvfile:
                    fieldnames = [
                        "step", "sim_time", "plan_time", "qpos", "qvel",
                        "control", "running_cost", # "terminal_cost"
                    ]
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    for log in logs:
                        writer.writerow(log)
                print(f"Logs saved to {log_file}")
