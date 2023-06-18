CUDA_VISIBLE_DEVICES=0 python -m mbrl.examples.main seed=0 exp_name=walker_run  is_mpc=1  algorithm=mbpo  overrides=mbpo_walker_walk  dynamics_model.activation_fn_cfg._target_=torch.nn.ReLU

