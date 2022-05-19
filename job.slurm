#!/bin/bash

#SBATCH --job-name=comb_1_20    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=48:05:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=mxs2361@mavs.uta.edu
#SBATCH -o slurm_comb_2_output/gan_output-%j


source ~/miniconda3/etc/profile.d/conda.sh
conda init bash
which python
conda activate pytorch_gpu

# have to change tiles, datasource, codex_data_module

python3 train_lightning.py --saturation_clip_input False \
--saturation_clip_target False --data_type 16 --batch_size 4 --n_epochs 250 \
--n_downsample 2 --n_residual 2 --n_df 10 --n_D 3 \
--tb_logger_name combination_2 --dataset_name combination_v2_31 \
--raw_data_dir /home/mxs2361/Dataset/codex_data/raw_data_scaled/ \
--channel_ids channel_id_distribution/combinations/h_to_nih_train_comb_31.json
