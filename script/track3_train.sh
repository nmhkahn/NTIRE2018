python track23/train.py \
                    --patch_size 48 \
                    --lr 0.0001 \
                    --batch_size 64 16 8 \
                    --max_steps 25000 25000 100000 \
                    --data_path dataset/DIV2K/div2k_train.h5 \
                    --data_names x4d x4 x2 HR \
                    --scales 4 4 2 1 \
                    --model progressive_carn \
                    --ckpt_name progressive_carn \
                    --print_every 500 \
                    --num_gpu 2 \
                    --test_dirname dataset/DIV2K/valid \
                    --test_data_from DIV2K_valid_LR_x4d \
                    --test_data_to DIV2K_valid_HR \

