#!/bin/bash

#SBATCH --job-name=lightning_gan    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=256G
#SBATCH --gres=gpu:2             # number of gpus per node
#SBATCH --time=48:05:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=mxs2361@mavs.uta.edu
#SBATCH -o slurm_output/gan_output-%j


source ~/miniconda3/etc/profile.d/conda.sh
conda init bash
which python
conda activate pytorch_gpu


python3 train_lightning.py --input_ch 20 --target_ch 9 --input_dir_train \
/home/mxs2361/Dataset/codex_data/Data_scaled_20_9/train_A/ --target_dir_train \
/home/mxs2361/Dataset/codex_data/Data_scaled_20_9/train_B/  --n_workers 256 --saturation_clip_input False \
--saturation_clip_target False --data_type 4 --batch_size 4 --n_epochs 250 \
--n_downsample 2 --n_residual 2 --n_df 10 --n_D 3 \
--tb_logger_name four_channel_20_9 --dataset_name four_channel_20_9

