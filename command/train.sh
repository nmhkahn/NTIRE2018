python track1/train.py \
                    --patch_size 48 \
                    --lr 0.0001 \
                    --decay 100000 \
                    --batch_size 32 4 1 \
                    --max_steps 200000 200000 400000 \
                    --data_path dataset/DIV2K/div2k_train.h5 \
                    --data_names x8 x4 x2 HR\
                    --scales 8 4 2 1 \
                    --model progressive_carn \
                    --ckpt_name progressive_carn \
                    --print_every 1000 \
                    --num_gpu 1 \
                    --test_dirname dataset/DIV2K/valid \
                    --test_data_from DIV2K_valid_LR_x8 \
                    --test_data_to DIV2K_valid_HR \

