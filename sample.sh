python src/sample.py --model=carn \
                     --ckpt_path=checkpoint/carn_8_1_50000.pth \
                     --dirname=dataset/DIV2K/valid \
		        	 --data_from=DIV2K_valid_LR_x8 \
			         --data_to=DIV2K_valid_HR \
                     --act=relu \
                     --sample_dir=result
