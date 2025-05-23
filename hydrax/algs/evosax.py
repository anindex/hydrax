from typing import Any, Tuple

from evosax.algorithms.population_based.base import PopulationBasedAlgorithm
from evosax.algorithms.distribution_based.base import DistributionBasedAlgorithm
import jax
import jax.numpy as jnp
from flax.struct import dataclass

from hydrax.alg_base import SamplingBasedController, Trajectory
from hydrax.risk import RiskStrategy
from hydrax.task_base import Task

# Generic types for evosax
EvoParams = Any
EvoState = Any


@dataclass
class EvosaxParams:
    """Policy parameters for evosax optimizers.

    Attributes:
        controls: The latest control sequence, U = [u₀, u₁, ..., ].
        opt_state: The state of the evosax optimizer (covariance, etc.).
        rng: The pseudo-random number generator key.
    """

    controls: jax.Array
    opt_state: EvoState
    rng: jax.Array


class Evosax(SamplingBasedController):
    """A generic controller that allows us to use any evosax optimizer.

    See https://github.com/RobertTLange/evosax/ for details and a list of
    available optimizers.
    """

    def __init__(
        self,
        task: Task,
        optimizer: PopulationBasedAlgorithm | DistributionBasedAlgorithm,
        num_samples: int,
        es_params: EvoParams = None,
        num_randomizations: int = 1,
        risk_strategy: RiskStrategy = None,
        seed: int = 0,
        **kwargs,
    ):
        """Initialize the controller.

        Args:
            task: The dynamics and cost for the system we want to control.
            optimizer: The evosax optimizer to use.
            num_samples: The number of control tapes to sample.
            es_params: The parameters for the evosax optimizer.
            num_randomizations: The number of domain randomizations to use.
            risk_strategy: How to combining costs from different randomizations.
                           Defaults to average cost.
            seed: The random seed for domain randomization.
            **kwargs: Additional keyword arguments for the optimizer.
        """
        super().__init__(task, num_randomizations, risk_strategy, seed)

        self.strategy = optimizer(
            population_size=num_samples,
            solution=jnp.zeros(
                (task.planning_horizon * task.model.nu,)
            ),
            **kwargs,
        )

        if es_params is None:
            es_params = self.strategy.default_params
        self.es_params = es_params

    def init_params(self, seed: int = 0) -> EvosaxParams:
        """Initialize the policy parameters."""
        rng = jax.random.key(seed)
        rng, init_rng = jax.random.split(rng)
        controls = jnp.zeros((self.task.planning_horizon, self.task.model.nu))
        dummy_solution = jnp.zeros(
            (self.task.planning_horizon * self.task.model.nu,)
        )
        if isinstance(self.strategy, DistributionBasedAlgorithm):
            opt_state = self.strategy.init(init_rng, dummy_solution, self.es_params)
        else:
            fitness = jnp.full((self.strategy.population_size,), jnp.inf)
            dummy_solution = dummy_solution[None, :].repeat(
                self.strategy.population_size, axis=0
            )
            opt_state = self.strategy.init(init_rng, dummy_solution, fitness, self.es_params)
        return EvosaxParams(controls=controls, opt_state=opt_state, rng=rng)

    def sample_controls(
        self, params: EvosaxParams
    ) -> Tuple[jax.Array, EvosaxParams]:
        """Sample control sequences from the proposal distribution."""
        rng, sample_rng = jax.random.split(params.rng)
        x, opt_state = self.strategy.ask(
            sample_rng, params.opt_state, self.es_params
        )

        # evosax works with vectors of decision variables, so we reshape U to
        # [batch_size, horizon, nu].
        controls = jnp.reshape(
            x,
            (
                self.strategy.population_size,
                self.task.planning_horizon,
                self.task.model.nu,
            ),
        )
        controls = jnp.nan_to_num(controls)  # avoid NaNs in the controls

        return controls, params.replace(opt_state=opt_state, rng=rng)

    def update_params(
        self, params: EvosaxParams, rollouts: Trajectory
    ) -> EvosaxParams:
        """Update the policy parameters based on the rollouts."""
        rng, sample_rng = jax.random.split(params.rng)
        costs = jnp.sum(rollouts.costs, axis=1)  # sum over time steps
        x = jnp.reshape(rollouts.controls, (self.strategy.population_size, -1))
        opt_state, _ = self.strategy.tell(
            sample_rng, x, costs, params.opt_state, self.es_params
        )

        best_idx = jnp.argmin(costs)
        best_controls = rollouts.controls[best_idx]

        # By default, opt_state stores the best member ever, rather than the
        # best member from the current generation. We want to just use the best
        # member from this generation, since the cost landscape is constantly
        # changing.
        opt_state = opt_state.replace(
            best_solution=x[best_idx], best_fitness=costs[best_idx]
        )

        return params.replace(rng=rng, controls=best_controls, opt_state=opt_state)

    def get_action(self, params: EvosaxParams, t: float) -> jax.Array:
        """Get the control action for the current time step, zero order hold."""
        idx_float = t / self.task.dt  # zero order hold
        idx = jnp.floor(idx_float).astype(jnp.int32)
        return params.controls[idx]
