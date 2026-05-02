


# Uncomment the following script for training step 0.
# CUDA_VISIBLE_DEVICES=0 nohup python -u train.py \
#         --cfg-path config_llama_3.2_only_proj.yaml \
#         --start_train_step 0 > order_1_step_0.log 2>&1 &


# Uncomment the following script for training step 1.
# CUDA_VISIBLE_DEVICES=0 nohup python -u train.py \
#         --cfg-path config_llama_3.2_only_proj.yaml \
#         --start_train_step 1 \
#         --text_instruct_num 4 \
#         --text_instruct_constraint_type KL \
#         --alpha_fusion 0.1 > order_1_step_1.log 2>&1 &


# Uncomment the following script for training step 2.
# CUDA_VISIBLE_DEVICES=0 nohup python -u train.py \
#         --cfg-path config_llama_3.2_only_proj.yaml \
#         --start_train_step 2 \
#         --text_instruct_num 4 \
#         --text_instruct_constraint_type KL \
#         --alpha_fusion 0.1 > order_1_step_2.log 2>&1 &


# Uncomment the following script for training step 3.
# CUDA_VISIBLE_DEVICES=0 nohup python -u train.py \
#         --cfg-path config_llama_3.2_only_proj.yaml \
#         --start_train_step 3 \
#         --text_instruct_num 4 \
#         --text_instruct_constraint_type KL \
#         --alpha_fusion 0.1 \
#         --pseudo \
#         --pseudo_num 4 \
#         --lambda_kl 1.0 > order_1_step_3.log 2>&1 &


# Uncomment the following script for training step 4.
# CUDA_VISIBLE_DEVICES=0 nohup python -u train.py \
#         --cfg-path config_llama_3.2_only_proj.yaml \
#         --start_train_step 4 \
#         --text_instruct_num 4 \
#         --text_instruct_constraint_type KL \
#         --alpha_fusion 0.5 \
#         --pseudo \
#         --pseudo_num 1 \
#         --lambda_kl 0.5 > order_1_step_4.log 2>&1 &


# Uncomment the following script for training step 5.
# CUDA_VISIBLE_DEVICES=0 nohup python -u train.py \
#         --cfg-path config_llama_3.2_only_proj.yaml \
#         --start_train_step 5 \
#         --text_instruct_num 4 \
#         --text_instruct_constraint_type KL \
#         --alpha_fusion 0.5 \
#         --pseudo \
#         --pseudo_num 4 \
#         --lambda_kl 0.1 > order_1_step_5.log 2>&1 &

