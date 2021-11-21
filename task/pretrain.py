from argparse import Namespace
import time
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor

from method.data.datamodule import ZincDataModule
from method.model.lightningmodule import MyLightningModule

def pretrain_model(args: Namespace):
    """
    The called pretrain function.

    :param args: arguments
    """

    dm = ZincDataModule.from_argparse_args(args)

    model = MyLightningModule(
            n_layers=args.n_layers,
            num_heads=args.num_heads,
            hidden_dim=args.hidden_dim,
            attention_dropout_rate=args.attention_dropout_rate,
            dropout_rate=args.dropout_rate,
            intput_dropout_rate=args.intput_dropout_rate,
            weight_decay=args.weight_decay,
            ffn_dim=args.ffn_dim,
            warmup_updates=args.warmup_updates,
            tot_updates=args.tot_updates,
            peak_lr=args.peak_lr,
            end_lr=args.end_lr,
            edge_type=args.edge_type,
            multi_hop_max_dist=args.multi_hop_max_dist,
            flag=args.flag,
            flag_m=args.flag_m,
            flag_step_size=args.flag_step_size,
            num_class=args.num_class
        )

    trainer = pl.Trainer.from_argparse_args(args)
    trainer.callbacks.append(LearningRateMonitor(logging_interval='step'))
    
    trainer.fit(model, datamodule=dm)
    
    result = trainer.test(datamodule=dm)