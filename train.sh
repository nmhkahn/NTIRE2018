python src/train.py \
                    --patch_size 32 \
                    --batch_size 64  16 8 \
                    --max_steps 100000 200000 400000 \
                    --data_path dataset/DIV2K/div2k_train.h5 \
                    --data_names x8 x4 x2 HR\
                    --scale 8 4 2 1 \
                    --model carn_prog \
                    --ckpt_name carn_prog \
                    --print_every 1000 \
                    --num_gpu 1 \
                    --test_dirname dataset/DIV2K/valid \
                    --test_data_from DIV2K_valid_LR_x8 \
                    --test_data_to DIV2K_valid_HR \

