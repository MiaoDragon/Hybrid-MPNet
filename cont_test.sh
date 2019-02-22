# reservoid with replay
# s2d
#python3 cmpnet_test.py --model_path ../cmpnet_result/res_replay/s2d/ \
#--no_env 100 --no_motion_paths 4000 --grad_step 1 --learning_rate 0.01 \
#--memory_strength 0.5 --n_memories 10000 \
#--n_tasks 1 --device 1 --data_path ../data/simple/ \
#--start_epoch 1 --memory_type res --env_type s2d
# c2d
python3 cont_test.py --model_path ../cont_results/cont_train_global/200-pretrain/ \
--no_env 100 --no_motion_paths 4000 --grad_step 1 --learning_rate 0.01 \
--memory_strength 0.5 --n_memories 10000 \
--n_tasks 1 --device 3 --data_path ../data/simple/ \
--start_epoch 1 --memory_type res --env_type s2d
