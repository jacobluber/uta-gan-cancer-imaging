from gc import callbacks
import json
from codex_data_module import CODEXDataModule
from pix2pixhd_lightning import Pix2PixHDCodex
from Pix2PixHD_Options import TrainOption
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import time


import pytorch_lightning as pl



if __name__ == '__main__':
    start_time = time.time()
    opt = TrainOption().parse()
    print(opt)
    with open(opt.channel_ids, 'r') as f:
        channel_ids = json.load(f)
    opt.input_ch = len(channel_ids['source_channel_ids'])
    opt.target_ch = len(channel_ids['target_channel_ids']) 

    data_mode = 'raw_data'
    tiles = False
    
    data = CODEXDataModule(src_data_dir = opt.input_dir_train, \
    tgt_data_dir = opt.target_dir_train, src_ch = len(channel_ids['source_channel_ids']), tgt_ch = len(channel_ids['target_channel_ids']), \
        src_channel_ids=channel_ids['source_channel_ids'], tgt_channel_ids=channel_ids['target_channel_ids'],
        raw_data_dir=opt.raw_data_dir, data_mode= data_mode, tiles = False)
    
    

    
    data.prepare_data()
    train_dataloader = data.train_dataloader()
    val_dataloader = data.val_dataloader()
    test_dataloader = data.test_dataloader()

    
    display_step = 1
    opt.dataset_name = opt.dataset_name + '_' +str(channel_ids['uid']) #For saving models and test images
    pix2pix = Pix2PixHDCodex(opt, display_step)
    logger = TensorBoardLogger("tb_logs_combinations_2", name= opt.tb_logger_name  + '_' + str(channel_ids['uid']))
    checkpoint_callback = ModelCheckpoint(dirpath="checkpoints/" + opt.dataset_name  + '/Model', save_top_k=2, monitor="Generator (val) Loss")
    # trainer = pl.Trainer(max_epochs=1000, gpus=-1, logger = logger)

    trainer = pl.Trainer(max_epochs=2500, precision=16, gpus = 1, min_epochs=100, auto_lr_find=True, auto_scale_batch_size=False,
                          progress_bar_refresh_rate=10, log_every_n_steps=1, \
                              check_val_every_n_epoch=10, \
                           logger = logger, callbacks = checkpoint_callback)#strategy="dp" #Data Parallel (strategy='dp') (multiple-gpus, 1 machine), tried strategy= 'ddp_spawn',
    trainer.fit(pix2pix, train_dataloader, val_dataloader)

    trainer.test(dataloaders=test_dataloader)
    end_time = time.time()
    print('Total elapsed time ', end_time - start_time)
