from codex_data_module import CODEXDataModule
from pix2pixhd_lightning import Pix2PixHDCodex
from Pix2PixHD_Options import TrainOption
from pytorch_lightning.loggers import TensorBoardLogger
import json


import pytorch_lightning as pl



if __name__ == '__main__':
    opt = TrainOption().parse()
    print(opt)
    with open(opt.channel_ids, 'r') as f:
        channel_ids = json.load(f)
    opt.input_ch = len(channel_ids['source_channel_ids'])
    opt.target_ch = len(channel_ids['target_channel_ids']) 

    data = CODEXDataModule(src_data_dir = opt.input_dir_train, \
    tgt_data_dir = opt.target_dir_train, src_ch = len(channel_ids['source_channel_ids']), tgt_ch = len(channel_ids['target_channel_ids']), \
        src_channel_ids=channel_ids['source_channel_ids'], tgt_channel_ids=channel_ids['target_channel_ids'],
        raw_data_dir=opt.raw_data_dir, data_mode='raw_data', tiles=False)
    
    

    
    data.prepare_data()
    train_dataloader = data.train_dataloader()
    val_dataloader = data.val_dataloader()
    test_dataloader = data.test_dataloader(test_mode = False)

    
    display_step = 1
    opt.dataset_name = opt.dataset_name + '_' +str(channel_ids['uid']) #For saving models and test images
    pix2pix = Pix2PixHDCodex(opt, display_step)
    logger = TensorBoardLogger("tb_logs_cluster_check", name= opt.tb_logger_name  + '_' + str(channel_ids['uid']))
    # trainer = pl.Trainer(max_epochs=1000, gpus=-1, logger = logger)

    trainer = pl.Trainer(max_epochs=1000, precision=16, gpus = 1, min_epochs=100, auto_lr_find=True, auto_scale_batch_size=False,
                          progress_bar_refresh_rate=10, log_every_n_steps=1, \
                              check_val_every_n_epoch=10, \
                           logger = logger)#strategy="dp" #Data Parallel (strategy='dp') (multiple-gpus, 1 machine), tried strategy= 'ddp_spawn',

    #pix2pix = Pix2PixHDCodex.load_from_checkpoint('checkpoints/cross_dataset_and_dapi_cross_data_check_6_2_and_DAPI/Model/epoch=9-step=40.ckpt')
    #checkpoints/cross_dataset_and_dapi_cross_data_check_6_2_and_DAPI/Model/epoch=319-step=119040.ckpt
    #Below one is for tiles based model
    #Trained patch : checkpoints/cross_dataset_and_dapi_cross_data_check_6_2_and_DAPI/Model/epoch=659-step=245520.ckpt
    model_dir = opt.model_dir
    trained_model_dir = opt.trained_model_dir 
    print(trained_model_dir * 10)
    pix2pix = Pix2PixHDCodex.load_from_checkpoint(trained_model_dir)
    trainer.test(model = pix2pix, dataloaders=test_dataloader)
    # print(pix2pix(test_dataloader))


###
