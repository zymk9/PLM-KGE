from posixpath import commonpath
import torch
import json
import torch.backends.cudnn as cudnn

from doc import Dataset, RelGenDataset
from config import args
from trainer import Trainer
from logger_config import logger


def main():
    ngpus_per_node = torch.cuda.device_count()
    cudnn.benchmark = True

    logger.info("Use {} gpus for training".format(ngpus_per_node))

    trainer = Trainer(args, ngpus_per_node=ngpus_per_node)
    logger.info('Args={}'.format(json.dumps(args.__dict__, ensure_ascii=False, indent=4)))
    trainer.train_loop()


if __name__ == '__main__':
    ngpus_per_node = torch.cuda.device_count()
    trainer = Trainer(args, ngpus_per_node=ngpus_per_node)
    main()
    """
    train_dataset = RelGenDataset(path=args.train_path, task=args.task, batch_size = args.batch_size, commonsense_path = args.commonsense_path)
    mapping = {
      "1": [],
      "N": []
    }
    for i in range(200):
        print(train_dataset[i])
    """