CUDA_VISIBLE_DEVICES=0 python model/opt.py \
    facebook/opt-125m \
    --wbits 2 \
    --groupsize -1 \
    --acc \
    --tcq \
    --bcq_round 20 # bcq_round 20 works too, bigger - slower - maybe better