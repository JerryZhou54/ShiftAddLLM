CUDA_VISIBLE_DEVICES=0 python model/bloom.py \
    bigscience/bloom-7b1 \
    --wbits 3 \
    --groupsize -1 \
		--acc \
    --bcq_round 50 # bcq_round 20 works too, bigger - slower - maybe better