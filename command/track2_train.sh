python track234/train.py \
                    --patch_size 48 \
                    --lr 0.0001 \
                    --decay 100000 \
                    --batch_size 32 8 \
                    --max_steps 200000 200000 \
                    --data_path dataset/DIV2K/div2k_train.h5 \
                    --data_names x4m x4 HR \
                    --scales 4 4 1 \
                    --model progressive_carn \
                    --ckpt_name progressive_carn \
                    --print_every 1000 \
                    --num_gpu 1 \
                    --test_dirname dataset/DIV2K/valid \
                    --test_data_from DIV2K_valid_LR_x4m \
                    --test_data_to DIV2K_valid_HR \

