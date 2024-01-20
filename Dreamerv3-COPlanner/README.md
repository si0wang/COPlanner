# Proprio DMC:

CUDA_VISIBLE_DEVICES=0 python dreamerv3/train.py --seed 0 --configs dmc_proprio --task dmc_hopper_hop --exp_name coplanner_hopper_hop_c05o1 --is_mpc 1 --conservative_rate 0.5 --optimistic_rate 1 --logdir ./coplanner_dreamerv3_prop_exp


# Visual DMC

CUDA_VISIBLE_DEVICES=0 python dreamerv3/train.py --seed 0 --configs dmc_vision --task dmc_cartpole_swingup_sparse --is_mpc 1 --conservative_rate 0.5 --optimistic_rate 1 --exp_name coplanner_cartpole_swingup_sparse_c05o1 --logdir ./coplanner_dreamerv3_exp
