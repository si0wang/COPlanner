CUDA_VISIBLE_DEVICES=0 python3 dreamer.py --configs defaults dmc --task dmc_hopper_hop --exp_name hopper_hop --seed 1 --optimistic_rate 0.1 --conservative_rate 1 --expl_behavior plan2explore

