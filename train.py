
import argparse
import os
import numpy as np
import random
import time
import datetime
from pathlib import Path


from core.soundstream import SoundStream
from trainer.trainer import SoundStreamTrainer



def main(args):
    
    soundstream = SoundStream(
        codebook_dim = args.codebook_dim,
        codebook_size = args.codebook_size,
        rq_num_quantizers = args.rq_num_quantizers,
        use_local_attn = True,
        use_mhesa = False,
        attn_window_size = args.attn_window_size,       # local attention receptive field at bottleneck
        attn_depth = args.attn_depth                # 2 local attention transformer blocks - the soundstream folks were not experts with attention, so i took the liberty to add some. encodec went with lstms, but attention should be better
    )

    trainer = SoundStreamTrainer(
    soundstream,
    folder = args.folder,
    batch_size = args.batch_size,
    grad_accum_every = args.grad_accum_every,         # effective batch size of 32
    data_max_length_seconds = args.data_max_length_seconds,  # train on 2 second audio
    num_train_steps = args.num_train_steps,
    results_folder = args.output_dir,
    
    save_results_every = args.save_results_every,
    save_model_every = args.save_model_every,
    log_losses_every = args.log_losses_every,
    ).cuda()

    print("Starting training")

    trainer.train()
     





if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', default="/srv/share4/sanisetty3/MagnaTagATune/data" , help="folder with train and test data")
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--output_dir', default="/srv/scratch/sanisetty3/soundstream_music/checkpoints/no_deepspeed/fixed_input_length_no_att/")
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--codebook_dim', default=768, type=int)
    parser.add_argument('--codebook_size', default=1024, type=int)
    parser.add_argument('--rq_num_quantizers', default=8, type=int)
    parser.add_argument('--attn_window_size', default=128, type=int)
    parser.add_argument('--attn_depth', default=2, type=int,)
    parser.add_argument('--batch_size', default=2, type=int,)
    parser.add_argument('--data_max_length_seconds', default=2, type=int,)
    parser.add_argument('--grad_accum_every', default=8, type=int)
    parser.add_argument("--num_train_steps",  default=10000,type=int)
    parser.add_argument("--save_results_every",  default=100,type=int)
    parser.add_argument("--save_model_every",  default=1000,type=int)
    parser.add_argument("--log_losses_every",  default=1,type=int)

    args = parser.parse_args()

    import torch
    torch.cuda.empty_cache()

    
    main(args)









# accelerate configuration saved at /nethome/sanisetty3/.cache/huggingface/accelerate/default_config.yaml   