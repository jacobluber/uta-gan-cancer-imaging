from gc import callbacks
from codex_data_module import CODEXDataModule
from pix2pixhd_lightning import Pix2PixHDCodex
from Pix2PixHD_Options import TrainOption
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint



import pytorch_lightning as pl



if __name__ == '__main__':
    opt = TrainOption().parse()
    print(opt)
    data = CODEXDataModule(src_data_dir = opt.input_dir_train, \
    tgt_data_dir = opt.target_dir_train, src_ch = opt.input_ch, tgt_ch = opt.target_ch, \
        raw_data_dir=opt.raw_data_dir, data_mode='raw_data')
    data.prepare_data()
    train_dataloader = data.train_dataloader()
    val_dataloader = data.val_dataloader()
    test_dataloader = data.test_dataloader()

    
    display_step = 1
    
    pix2pix = Pix2PixHDCodex(opt, display_step)
    logger = TensorBoardLogger("tb_logs", name= opt.tb_logger_name)
    checkpoint_callback = ModelCheckpoint(dirpath="checkpoints/" + opt.dataset_name + '/Model', save_top_k=2, monitor="Generator (val) Loss")
    # trainer = pl.Trainer(max_epochs=1000, gpus=-1, logger = logger)

    trainer = pl.Trainer(max_epochs=100, precision=16, gpus = 1, min_epochs=100, auto_lr_find=True, auto_scale_batch_size=False,
                          progress_bar_refresh_rate=10, log_every_n_steps=1, \
                              check_val_every_n_epoch=10, \
                           logger = logger, callbacks = checkpoint_callback)#strategy="dp" #Data Parallel (strategy='dp') (multiple-gpus, 1 machine), tried strategy= 'ddp_spawn',
    trainer.fit(pix2pix, train_dataloader, val_dataloader)

    trainer.test(dataloaders=test_dataloader)