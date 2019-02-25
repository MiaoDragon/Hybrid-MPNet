python3 hybrid_train_global_demo.py --model_path ../hybrid_res/global/r3d/no_include/ --AEtype_deep 1 \
--no_env 2 --no_motion_paths 2 --grad_step 1 \
--num_epochs 1 --memory_strength 0.5 --n_memories 10000 \
--n_tasks 1 --device 0 --freq_rehersal 100 --batch_rehersal 100 \
--start_epoch 0 --MAX_NEURAL_REPLAN 11 --pretrain_path 200 --data_path /media/arclabdl1/HD1/Ahmed/r-3d/dataset2/ \
--include_suc_path 0 --world_size 20 --env_type r3d \
--total_input_size 6006 --AE_input_size 6000 --mlp_input_size 66 --output_size 3

# 100 x 4000
