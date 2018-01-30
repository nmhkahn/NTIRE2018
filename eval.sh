python src/eval.py --model=carn_large \
                     --ckpt_path=checkpoint/carn_large_8_1_222000.pth \
                     --dirname=dataset/DIV2K/valid \
		        	 --data_from=DIV2K_valid_LR_x8 \
			         --data_to=DIV2K_valid_HR \
                     --act=relu \
