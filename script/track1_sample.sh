python track1/sample.py \
    --model progressive_carn \
    --dirname dataset/DIV2K/test \
    --data_from DIV2K_test_LR_x8 \
    --data_to DIV2K_test_HR \
    --chunk 2 \
    --sample_dir result \
    --ckpt_path checkpoint/best.pth \
    --stage 2
