import ray
from agamoo_ray.players import ClonalSelection
from agamoo_ray.objectives import RE31, RE32, RE33, RE34, RE35, RE36, RE37
from agamoo_ray import AGAMOO, Evaluator
import logging

logging.basicConfig(level=logging.INFO)

max_eval = 100000
npop = 25
change_iter = 1
next_iter = -1
max_front = 200
player_parm = { "nclone": 12, "mutate_args": [0.45, 0.9, 0.01], 'sup': 0.0}

obj1 = RE32(0, obj=1)
obj2 = RE32(1, obj=2)
obj3 = RE32(2, obj=3)

objs = [obj1, obj2, obj3]

nvar = obj1.n_var
nobj = obj1.n_obj

ray.init(log_to_driver=True, include_dashboard=False)

agamoo = AGAMOO(max_eval, change_iter, next_iter, max_front, verbose=True)
storage = agamoo.create_storage(nvar, nobj, num_cpus=1)
evaluator = Evaluator.options(num_cpus=0).remote(objs)
players = [ClonalSelection.options(num_cpus=1).remote(i, npop, player_parm, obj, storage, exchange='front_random', verbose=True) for i, obj in zip(range(nobj), objs)]
agamoo.init_players(players, evaluator)

agamoo.start_optimize()