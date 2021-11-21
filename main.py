import pytorch_lightning as pl

from method.utils.parsing import parse_args
from task.pretrain import pretrain_model
from task.cross_validate import cross_validate

if __name__ == '__main__':
    #setup(seed=45)

    args = parse_args()
    pl.seed_everything(args.seed)

    if args.task == 'pretrain':
        pretrain_model(args)

    if args.task == 'finetune':
        cross_validate(args)