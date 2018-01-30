CUDA_VISIBLE_DEVICES=1 python src/train.py \
                    --patch_size 16 \
                    --batch_size 32 \
                    --max_steps 300000 \
                    --decay 20000 \
                    --data_path dataset/DIV2K/div2k_train.h5 \
                    --data_names HR x2 x4 x8 \
                    --scale 1 2 4 8 \
                    --model carn_lp2 \
                    --ckpt_name carn_lp2 \
                    --print_every 10 \
                    --num_gpu 1 \
                    --test_dirname dataset/DIV2K/valid \
                    --test_data_from DIV2K_valid_LR_x8 \
                    --test_data_to DIV2K_valid_HR \

