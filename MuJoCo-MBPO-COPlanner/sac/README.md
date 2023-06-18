
CUDA_VISIBLE_DEVICES=0 nohup python main_sac_dmc.py --domain_name 'hopper' --task_name 'stand' --exp_name hopper_stand_sac_3  --alpha 0.05 --seed 3 > /cmlscratch/xywang/code/mbpo_mpcrollout/exp_mpc_rollout/exp20.log 2>&1 &
