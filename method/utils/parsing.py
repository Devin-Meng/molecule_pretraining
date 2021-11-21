from argparse import ArgumentParser
import pytorch_lightning as pl
from method.data.datamodule import ZincDataModule
from method.model.lightningmodule import MyLightningModule

def parse_args():
    """
    Parses args for all task

    :return: A Namespace
    """
    parser = ArgumentParser()
    parser.add_argument('task', type=str)
    parser.add_argument('--input_path', type=str)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = MyLightningModule.add_model_specific_args(parser)
    parser = ZincDataModule.add_argparse_args(parser)

    args = parser.parse_args()

    # if args.parser_name == 'finetune' or args.parser_name == 'eval':
    #     modify_train_args(args)
    # elif args.parser_name == "pretrain":
    #     modify_pretrain_args(args)
    # elif args.parser_name == 'predict':
    #     modify_predict_args(args)
    # elif args.parser_name == 'fingerprint':
    #     modify_fingerprint_args(args)

    return args