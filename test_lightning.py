from codex_data_module import CODEXDataModule
from pix2pixhd_lightning import Pix2PixHDCodex
from Pix2PixHD_Options import TrainOption
from pytorch_lightning.loggers import TensorBoardLogger



import pytorch_lightning as pl



if __name__ == '__main__':
    opt = TrainOption().parse()
    # print(opt)
    data = CODEXDataModule(src_data_dir = opt.input_dir_train, \
    tgt_data_dir = opt.target_dir_train, src_ch = opt.input_ch, tgt_ch = opt.target_ch, demo_data=False)
    data.prepare_data()
    val_dataloader = data.val_dataloader()
    test_dataloader = data.test_dataloader()

    
    display_step = 1
    
    # pix2pix = Pix2PixHDCodex(opt, display_step)
    pix2pix = Pix2PixHDCodex.load_from_checkpoint('tb_logs/four_channel_gray/version_2/checkpoints/epoch=999-step=16000.ckpt')
    logger = TensorBoardLogger("tb_logs", name= opt.tb_logger_name)
    # trainer = pl.Trainer(max_epochs=1000, gpus=-1, logger = logger)
    trainer = pl.Trainer(max_epochs=1000, gpus = 1, min_epochs=1, auto_lr_find=True, auto_scale_batch_size=False,
                          progress_bar_refresh_rate=10, log_every_n_steps=1, \
                              check_val_every_n_epoch=10, \
                           logger = logger)#strategy="dp" #Data Parallel (strategy='dp') (multiple-gpus, 1 machine), tried strategy= 'ddp_spawn',

    trainer.test(model = pix2pix, dataloaders=test_dataloader)
    # print(pix2pix(test_dataloader))