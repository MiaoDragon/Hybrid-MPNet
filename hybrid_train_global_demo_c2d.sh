python3 hybrid_train_global_demo.py --model_path ../hybrid_res/global/c2d/no_include/ \
--no_env 100 --no_motion_paths 4000 --grad_step 1 \
--num_epochs 1 --memory_strength 0.5 --n_memories 10000 \
--n_tasks 1 --device 1 --freq_rehersal 100 --batch_rehersal 100 \
--start_epoch 0 --MAX_NEURAL_REPLAN 11 --pretrain_path 200 --data_path /media/arclabdl1/HD1/Ahmed/r-2d/ \
--include_suc_path 0 --world_size 20 --env_type c2d \
--total_input_size 2804 --AE_input_size 2800 --mlp_input_size 32
