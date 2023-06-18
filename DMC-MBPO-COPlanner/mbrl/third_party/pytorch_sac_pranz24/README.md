CUDA_VISIBLE_DEVICES=0 nohup python main.py --seed 0  --domain_name 'hopper' --task_name 'stand' --exp_name hopper_hop_sac_2  --alpha 0.05 > /cmlscratch/xywang/code/mbpo_mpcrollout/exp_mpc_rollout/exp20.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python main.py --seed 3  --domain_name 'quadruped' --task_name 'walk' --exp_name quadruped_walk_sac_4  --alpha 0.05 > /cmlscratch/xywang/code/mbpo_mpcrollout/exp_mpc_rollout/exp20.log 2>&1 &

CUDA_VISIBLE_DEVICES=7 nohup python main.py --seed 3  --domain_name 'cheetah' --task_name 'run' --exp_name cheetah_run_sac_3  --alpha 0.05 > /cmlscratch/xywang/code/mbpo_mpcrollout/exp_mpc_rollout/exp20.log 2>&1 &

CUDA_VISIBLE_DEVICES=6 nohup python main.py --seed 10  --domain_name 'walker' --task_name 'run' --exp_name walker_run_sac_10  --alpha 0.05 > /cmlscratch/xywang/code/mbpo_mpcrollout/exp_mpc_rollout/exp20.log 2>&1 &

CUDA_VISIBLE_DEVICES=4 nohup python main.py --seed 0  --domain_name 'walker' --task_name 'walk' --exp_name walker_walk_sac_0  --alpha 0.05 > /cmlscratch/xywang/code/mbpo_mpcrollout/exp_mpc_rollout/exp20.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python main.py --seed 0  --domain_name 'walker' --task_name 'stand' --exp_name walker_stand_sac_0  --alpha 0.05 > /cmlscratch/xywang/code/mbpo_mpcrollout/exp_mpc_rollout/exp20.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python main.py --seed 3  --domain_name 'ball_in_cup' --task_name 'catch' --exp_name cup_catch_sac_3  --alpha 0.05 > /cmlscratch/xywang/code/mbpo_mpcrollout/exp_mpc_rollout/exp20.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python main.py --seed 0  --domain_name 'cartpole' --task_name 'swingup' --exp_name cartpole_swingup_sac_0  --alpha 0.05 > /cmlscratch/xywang/code/mbpo_mpcrollout/exp_mpc_rollout/exp20.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python main.py --seed 3  --domain_name 'finger' --task_name 'spin' --exp_name finger_spin_sac_3  --alpha 0.05 > /cmlscratch/xywang/code/mbpo_mpcrollout/exp_mpc_rollout/exp20.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python main.py --seed 3  --domain_name 'reacher' --task_name 'easy' --exp_name reacher_easy_sac_3  --alpha 0.05 > /cmlscratch/xywang/code/mbpo_mpcrollout/exp_mpc_rollout/exp20.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python main.py --seed 3  --domain_name 'reacher' --task_name 'hard' --exp_name reacher_hard_sac_3  --alpha 0.05 > /cmlscratch/xywang/code/mbpo_mpcrollout/exp_mpc_rollout/exp20.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python main.py --seed 3  --domain_name 'humanoid' --task_name 'walk' --num_steps 10000000 --exp_name humanoid_walk_sac_3  --alpha 0.05 > /cmlscratch/xywang/code/mbpo_mpcrollout/exp_mpc_rollout/exp20.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python main.py --seed 0  --domain_name 'humanoid' --task_name 'run' --num_steps 10000000 --exp_name humanoid_run_sac_0  --alpha 0.05 > /cmlscratch/xywang/code/mbpo_mpcrollout/exp_mpc_rollout/exp20.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python main.py --seed 3  --domain_name 'dog' --task_name 'walk' --num_steps 10000000 --exp_name dog_walk_sac_3  --alpha 0.05 > /cmlscratch/xywang/code/mbpo_mpcrollout/exp_mpc_rollout/exp20.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python main.py --seed 3  --domain_name 'manipulator' --task_name 'bring_ball' --num_steps 10000000 --exp_name manipulator_bring_ball_sac_3  --alpha 0.05 > /cmlscratch/xywang/code/mbpo_mpcrollout/exp_mpc_rollout/exp20.log 2>&1 &
