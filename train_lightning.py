from codex_data_module import CODEXDataModule
from pix2pixhd_lightning import Pix2PixHDCodex
from Pix2PixHD_Options import TrainOption
from pytorch_lightning.loggers import TensorBoardLogger



import pytorch_lightning as pl



if __name__ == '__main__':
    opt = TrainOption().parse()
    print(opt)
    data = CODEXDataModule(src_data_dir = opt.input_dir_train, \
    tgt_data_dir = opt.target_dir_train, demo_data=False)
    data.prepare_data()
    dataloader = data.train_dataloader()
    
    display_step = 1
    
    pix2pix = Pix2PixHDCodex(opt, display_step)
    logger = TensorBoardLogger("tb_logs", name="lightning_codex")
    # trainer = pl.Trainer(max_epochs=1000, gpus=-1, logger = logger)
    trainer = pl.Trainer(max_epochs=1000, gpus = 1, min_epochs=1, auto_lr_find=True, auto_scale_batch_size=False,
                          progress_bar_refresh_rate=10, log_every_n_steps=1, \
                           logger = logger)#strategy="dp" #Data Parallel (strategy='dp') (multiple-gpus, 1 machine), tried strategy= 'ddp_spawn',
    trainer.fit(pix2pix, dataloader)