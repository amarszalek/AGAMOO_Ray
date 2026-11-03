import inspect
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import ray

from agamoo_ray.player import Player
from agamoo_ray.objective import Objective

logger = logging.getLogger(__name__)


def _plain_class(cls: Any) -> Any:
    """
    Returns the undecorated class hidden behind a `@ray.remote` actor class.

    Sub-players of a MixPlayer run IN-PROCESS inside the MixPlayer actor, so they must be
    ordinary Python objects, not Ray actors. The library exports its players already
    decorated, hence the unwrapping.
    """
    # ray.remote(cls) -> ActorClass with __ray_metadata__.modified_class, which is a subclass
    # of the user's class carrying __ray_actor_class__ = the pristine original.
    meta = getattr(cls, '__ray_metadata__', None)
    inner = getattr(meta, 'modified_class', None)
    if inner is not None:
        return getattr(inner, '__ray_actor_class__', inner)
    inner = getattr(cls, '__ray_actor_class__', None)
    if inner is not None:
        return inner
    if inspect.isclass(cls):
        return cls
    raise TypeError(
        f"Cannot obtain a plain class from {cls!r}. Pass a player class (e.g. PSO), "
        f"or a factory callable returning a Player instance."
    )


@ray.remote
class MixPlayer(Player):
    """
    A player composed of several sub-players that take over one after another, each running
    for a declared share of this player's evaluation budget.

    The sub-players share one population, one Objective instance, one Repair and one tracker
    slot: only the `step()` implementation changes between phases. Everything the base Player
    does around `step()` - DVA gene masks, archive exchange, dispatching to GlobalStorage,
    heartbeats, environment changes - is untouched, so a MixPlayer is a drop-in replacement
    for any other player.

    player_param keys:
        players (Sequence): the sub-players, in the order they take over. Each entry is
            `(player_cls, player_param_dict)`, a bare `player_cls`, or a factory callable
            `f(num, npop, objective, storage_actor, gens, exchange, verbose, init_pop)`.
            Classes may be the ray-decorated ones exported by `agamoo_ray.players`.
        budget (Sequence[float]): share of the budget per sub-player, in the same order.
            Any positive scale works - [50, 50], [0.5, 0.5] and [1, 1] are equivalent;
            the values are normalized. A zero share disables that sub-player.
        progress (str): what the phase boundaries are measured against.
            'tracker' (default) - this tracker's entry in the global evaluation counter plus
                the evaluations not yet dispatched. Identical to the quantity the stop
                condition uses, so the phases span the run exactly.
            'own' - only the evaluations this player's own step() calls performed.
        max_eval (int, optional): budget the shares are taken from. None (default) = the
            global AGAMOO max_eval, read from the snapshot.
        rescale_budget (bool): if True (default), each sub-player sees a snapshot in which
            the evaluation counter and max_eval are rewritten to ITS phase, so a sub-player
            with an internal schedule (e.g. SimulatedAnnealing cooling) runs that schedule
            over its own phase instead of over the whole run.
        create (str): initial population strategy ('lhs' or 'uniform'). Defaults to the
            first sub-player's setting.

    Example:
        player_param = {
            'players': [(PSO, {'w': 0.729, 'c1': 1.49, 'c2': 1.49}),
                        (ClonalSelection, {'nclone': 12, 'strategy': 'base'})],
            'budget': [50, 50],
        }
        players = [MixPlayer.options(num_cpus=1).remote(i, npop, player_param, obj, storage,
                                                        gens='pattern', exchange='front_sup')
                   for i, obj in zip(range(nobj), objs)]
    """

    def __init__(self,
                 num: int,
                 npop: int,
                 player_param: Dict[str, Any],
                 objective: Objective,
                 storage_actor: Any,
                 gens: str = 'pattern',
                 exchange: str = 'front_sup',
                 verbose: bool = False,
                 init_pop: Optional[np.ndarray] = None):

        specs: Sequence[Any] = player_param.get('players', ())
        budget: Sequence[float] = player_param.get('budget', ())
        if len(specs) == 0:
            raise ValueError("MixPlayer needs at least one sub-player in player_param['players'].")
        if len(budget) != len(specs):
            raise ValueError(
                f"player_param['budget'] has {len(budget)} entries but "
                f"player_param['players'] has {len(specs)} - one share per sub-player is required."
            )

        shares = np.asarray(budget, dtype=float)
        if np.any(shares < 0):
            raise ValueError(f"Budget shares must be non-negative, got {list(budget)}.")
        total = float(shares.sum())
        if total <= 0:
            raise ValueError(f"Budget shares must sum to a positive value, got {list(budget)}.")
        shares = shares / total

        self.shares: np.ndarray = shares
        self._cum: np.ndarray = np.cumsum(shares)
        self.progress: str = player_param.get('progress', 'tracker')
        if self.progress not in ('tracker', 'own'):
            raise ValueError(f"Unknown progress mode {self.progress!r} (use 'tracker' or 'own').")
        self.max_eval: Optional[int] = player_param.get('max_eval', None)
        self.rescale_budget: bool = player_param.get('rescale_budget', True)

        # Sub-players are built BEFORE super().__init__ so that the default create_method can
        # be taken from the first one. They receive init_pop=None on purpose: the population
        # is owned by the MixPlayer and handed to step() as an argument.
        common = dict(gens=gens, exchange=exchange, verbose=verbose, init_pop=None)
        self.subs: List[Player] = [
            self._build_sub(spec, num, npop, objective, storage_actor, common) for spec in specs
        ]
        self.sub_names: List[str] = [type(s).__name__ for s in self.subs]

        create = player_param.get('create', None)
        if create is None:
            create = getattr(self.subs[0], 'create_method', 'lhs')
        self.create: str = create

        super().__init__(num, npop, objective, storage_actor, gens, exchange, verbose, init_pop,
                         create_method=self.create)

        self._phase_idx: int = int(np.searchsorted(self._cum, 0.0, side='right'))
        self._phase_idx = min(self._phase_idx, len(self.subs) - 1)
        self._own_evals: int = 0
        self._no_budget_warned: bool = False
        # (phase index, name, own evaluations at switch, iteration at switch)
        self.switch_log: List[Tuple[int, str, int, int]] = [
            (self._phase_idx, self.sub_names[self._phase_idx], 0, 0)
        ]

    # ------------------------------------------------------------------ construction

    @staticmethod
    def _build_sub(spec: Any, num: int, npop: int, objective: Objective,
                   storage_actor: Any, common: Dict[str, Any]) -> Player:
        """Instantiates one sub-player from a spec entry."""
        if isinstance(spec, (tuple, list)):
            if len(spec) != 2:
                raise ValueError(f"A sub-player spec tuple must be (class, param_dict), got {spec!r}.")
            target, param = spec[0], dict(spec[1] or {})
        else:
            target, param = spec, {}

        is_class_like = inspect.isclass(target) or hasattr(target, '__ray_metadata__')
        if not is_class_like:
            if not callable(target):
                raise TypeError(f"Sub-player spec {target!r} is neither a class nor a callable.")
            sub = target(num=num, npop=npop, objective=objective, storage_actor=storage_actor,
                         **common, **param)
        else:
            plain = _plain_class(target)
            params = inspect.signature(plain.__init__).parameters
            if 'player_param' in params:
                # PSO, ClonalSelection, CuckooSearch, SimulatedAnnealing, ...
                sub = plain(num, npop, param, objective, storage_actor, **common)
            else:
                # PymooPlayer and anything else taking its settings as keyword arguments
                sub = plain(num, npop, objective, storage_actor, **common, **param)

        if not isinstance(sub, Player):
            raise TypeError(f"{type(sub).__name__} is not a Player subclass.")
        return sub

    # ------------------------------------------------------------------ phase selection

    def _consumed_and_budget(self, global_state: Optional[Dict[str, Any]]) -> Tuple[float, Optional[float]]:
        """Evaluations spent so far and the budget the shares are taken from."""
        budget: Optional[float] = float(self.max_eval) if self.max_eval is not None else None

        if self.progress == 'own':
            consumed = float(self._own_evals)
        elif global_state is None:
            return 0.0, budget
        else:
            counters = global_state.get('evaluations_count')
            if counters is None:
                return 0.0, budget
            consumed = float(counters[self.tracker_idx]) + float(self.evaluation_counter)

        if budget is None and global_state is not None:
            glob = float(global_state.get('max_eval', 0) or 0)
            budget = glob if glob > 0 else None
        return consumed, budget

    def _select_phase(self, global_state: Optional[Dict[str, Any]]) -> int:
        consumed, budget = self._consumed_and_budget(global_state)
        if budget is None:
            if not self._no_budget_warned:
                logger.warning(
                    "MixPlayer %d has no evaluation budget (AGAMOO max_eval <= 0 and no "
                    "player_param['max_eval']), so it cannot switch phases and stays on %s.",
                    self.num, self.sub_names[self._phase_idx])
                self._no_budget_warned = True
            return self._phase_idx

        frac = min(consumed / budget, 1.0)
        idx = min(int(np.searchsorted(self._cum, frac, side='right')), len(self.subs) - 1)
        # Monotone: the counter can dip for a moment, because evaluation_counter is cleared at
        # dispatch slightly before GlobalStorage books it. Phases must never run backwards.
        return max(idx, self._phase_idx)

    def _sub_state(self, global_state: Optional[Dict[str, Any]], idx: int) -> Optional[Dict[str, Any]]:
        """
        Snapshot as seen by the active sub-player. With rescale_budget the evaluation counter
        and max_eval are rewritten so that the sub-player's own schedule spans its phase:
        it sees 0 at the start of its phase and its full budget at the end.
        """
        if global_state is None or not self.rescale_budget:
            return global_state
        consumed, budget = self._consumed_and_budget(global_state)
        if budget is None or 'evaluations_count' not in global_state:
            return global_state

        lo = float(self._cum[idx - 1]) * budget if idx > 0 else 0.0
        phase_budget = max((float(self._cum[idx]) * budget) - lo, 1.0)

        sub_state = dict(global_state)
        counters = np.array(global_state['evaluations_count'], dtype=float)
        # The sub-player adds its own evaluation_counter on top of this entry.
        counters[self.tracker_idx] = max(consumed - lo - float(self.evaluation_counter), 0.0)
        sub_state['evaluations_count'] = counters
        sub_state['max_eval'] = phase_budget
        return sub_state

    # ------------------------------------------------------------------ Player interface

    def step(self, pop: np.ndarray, pop_eval: np.ndarray, pattern: np.ndarray,
             global_state: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, np.ndarray, int]:
        """Delegates one evolutionary cycle to the sub-player owning the current phase."""
        idx = self._select_phase(global_state)
        if idx != self._phase_idx:
            self._phase_idx = idx
            self.switch_log.append((idx, self.sub_names[idx], self._own_evals, self.iteration))
            if self.verbose:
                logger.info("MixPlayer %d switched to %s at iteration %d (%d own evaluations).",
                            self.num, self.sub_names[idx], self.iteration, self._own_evals)

        sub = self.subs[idx]
        # The sub-player must act as if it were this player: same tracker slot, same pending
        # evaluation count (SimulatedAnnealing reads both), same repair, same iteration.
        sub.tracker_idx = self.tracker_idx
        sub.iteration = self.iteration
        sub.env_version = self.env_version
        sub.repair = self.repair
        sub.evaluation_counter = self.evaluation_counter

        pop, pop_eval, neval = sub.step(pop, pop_eval, pattern, self._sub_state(global_state, idx))
        self._own_evals += int(neval)
        return pop, pop_eval, neval

    def set_repair(self, repair: Any) -> None:
        super().set_repair(repair)
        for sub in self.subs:
            sub.repair = self.repair

    def update_environment(self, **kwargs) -> None:
        # The Objective instance is shared by every sub-player, so updating it once is enough;
        # only the version tags have to be kept in sync.
        super().update_environment(**kwargs)
        for sub in self.subs:
            sub.env_version = self.env_version

    def on_environment_change(self, pop: np.ndarray, pop_eval: np.ndarray) -> None:
        # Propagated to ALL sub-players, including those whose phase has not started yet:
        # fitness-dependent state (e.g. PSO pbest) must not survive an environment change.
        for sub in self.subs:
            sub.on_environment_change(pop, pop_eval)

    # ------------------------------------------------------------------ introspection

    def get_mix_info(self) -> Dict[str, Any]:
        """Phase layout and switch history, for post-run analysis."""
        return {
            'id': self.num,
            'criterion': self.objective.obj,
            'sub_players': list(self.sub_names),
            'shares': self.shares.tolist(),
            'boundaries': self._cum.tolist(),
            'phase': self._phase_idx,
            'active': self.sub_names[self._phase_idx],
            'own_evaluations': self._own_evals,
            'switches': [{'phase': p, 'player': n, 'own_evaluations': e, 'iteration': i}
                         for p, n, e, i in self.switch_log],
        }

