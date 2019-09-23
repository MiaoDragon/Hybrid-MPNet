cd ..
python3 hybrid_train_global_demo.py --model_path /media/arclabdl1/HD1/YLmiao/results/hybrid_res/global/s2d/ --AEtype_deep 1 \
--no_env 10 --no_motion_paths 40 --grad_step 1 \
--num_epochs 1 --memory_strength 0.5 --n_memories 10000 \
--n_tasks 1 --device 0 --freq_rehersal 100 --batch_rehersal 100 \
--start_epoch 0 --MAX_NEURAL_REPLAN 11 --pretrain_path 200 --data_path /media/arclabdl1/HD1/YLmiao/mpnet/data/simple/ \
--include_suc_path 0 --world_size 20 --env_type s2d \
--total_input_size 2804 --AE_input_size 2800 --mlp_input_size 32
cd exp
#100 x 4000
