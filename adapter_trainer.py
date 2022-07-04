from ast import arg
import glob
import json
import torch
import shutil
import os

import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from torch.nn import DataParallel

from typing import Dict
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers import AdamW

from doc import TripletDataset, Dataset, collate
from utils import AverageMeter, ProgressMeter
from utils import save_checkpoint, delete_old_ckt, report_num_trainable_parameters, move_to_cuda, get_model_obj, save_adapter
from metric import accuracy
from models import build_model, ModelOutput
from dict_hub import build_tokenizer
from logger_config import logger
from predict import BertPredictor
from evaluate import eval_single_direction
from dict_hub import get_entity_dict


class AdapterTrainer:

    def __init__(self, args, ngpus_per_node):
        self.args = args
        self.ngpus_per_node = ngpus_per_node
        build_tokenizer(args)

        # create model
        logger.info("=> creating model")
        self.model = build_model(self.args)
        logger.info(self.model)
        self._setup_training()

        self.optimizer = AdamW([p for p in self.model.parameters() if p.requires_grad],
                               lr=args.lr,
                               weight_decay=args.weight_decay)
        report_num_trainable_parameters(self.model)

        train_dataset_head = TripletDataset(path=args.train_path, mode='head-batch', ns_size=args.ns_size)
        train_dataset_tail = TripletDataset(path=args.train_path, mode='tail-batch', ns_size=args.ns_size)

        num_training_steps = args.epochs * len(train_dataset_head) // max(args.batch_size, 1) * 2 # 2 for alternating between dataloaders
        args.warmup = min(args.warmup, num_training_steps // 10)
        logger.info('Total training steps: {}, warmup steps: {}'.format(num_training_steps, args.warmup))
        self.scheduler = self._create_lr_scheduler(num_training_steps)
        self.best_metric = None

        self.train_loader_head = torch.utils.data.DataLoader(
            train_dataset_head,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=train_dataset_head.collate_fn,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True)

        self.train_loader_tail = torch.utils.data.DataLoader(
            train_dataset_tail,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=train_dataset_tail.collate_fn,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True)

        assert len(train_dataset_head) == len(train_dataset_tail)

    def train_loop(self):
        if self.args.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.epochs):
            # train for one epoch
            self.train_epoch(epoch)
            self._run_eval(epoch=epoch)

    @torch.no_grad()
    def _run_eval(self, epoch, step=0):
        metric_dict = self.eval_epoch(epoch)
        is_best = (self.best_metric is None or metric_dict['hit@1'] > self.best_metric['hit@1'])
        if is_best:
            self.best_metric = metric_dict

        filename = '{}/checkpoint_{}_{}.mdl'.format(self.args.model_dir, epoch, step)
        if step == 0:
            filename = '{}/checkpoint_epoch{}.mdl'.format(self.args.model_dir, epoch)
        save_checkpoint({
            'epoch': epoch,
            'args': self.args.__dict__,
            'state_dict': self.model.state_dict(),
        }, is_best=is_best, filename=filename)
        delete_old_ckt(path_pattern='{}/checkpoint_*.mdl'.format(self.args.model_dir),
                       keep=self.args.max_to_keep)

        hr_bert = self.model.hr_bert if not isinstance(self.model, DataParallel) else self.model.module.hr_bert
        tail_bert = self.model.tail_bert if not isinstance(self.model, DataParallel) else self.model.module.tail_bert

        hr_adapter = os.path.join(self.args.model_dir, f'link_pred_hr_epoch{epoch}')
        save_adapter(hr_bert, is_best, 'link_pred_hr', hr_adapter)

        tail_adapter = os.path.join(self.args.model_dir, f'link_pred_t_epoch{epoch}')
        save_adapter(tail_bert, is_best, 'link_pred_t', tail_adapter)


    @torch.no_grad()
    def eval_epoch(self, epoch) -> Dict:
        predictor = BertPredictor()
        predictor.load_existing_model(self.model)
        entity_dict = get_entity_dict()
        entity_tensor = predictor.predict_by_entities(entity_dict.entity_exs)

        forward_metrics = eval_single_direction(predictor,
                                                entity_tensor=entity_tensor,
                                                eval_forward=True,
                                                batch_size=self.args.batch_size,
                                                output_result=False)
        backward_metrics = eval_single_direction(predictor,
                                                entity_tensor=entity_tensor,
                                                eval_forward=False,
                                                batch_size=self.args.batch_size,
                                                output_result=False)
        metrics = {k: round((forward_metrics[k] + backward_metrics[k]) / 2, 4) for k in forward_metrics}
        logger.info('Epoch {}, valid metric: {}'.format(epoch, metrics))
        return metrics

    def train_epoch(self, epoch):
        losses = AverageMeter('Loss', ':.4')
        pos_losses = AverageMeter('Pos Loss', ':.4')
        neg_losses = AverageMeter('Neg Loss', ':.4')
        # hr_mean_norm = AverageMeter('HR mean norm', ':6.2f')
        # tail_mean_norm = AverageMeter('Tail mean norm', ':6.2f')

        progress = ProgressMeter(
            len(self.train_loader_head),
            [losses, pos_losses, neg_losses],
            prefix="Epoch: [{}]".format(epoch))

        for i, (batch_dict_head, batch_dict_tail) in enumerate(zip(self.train_loader_head, self.train_loader_tail)):
            self.train_step(batch_dict_head, losses, pos_losses, neg_losses)
            self.train_step(batch_dict_tail, losses, pos_losses, neg_losses)

            if i % self.args.print_freq == 0:
                progress.display(i)
            if (i + 1) % self.args.eval_every_n_step == 0:
                self._run_eval(epoch=epoch, step=i + 1)

        logger.info('Learning rate: {}'.format(self.scheduler.get_last_lr()[0]))

    def train_step(self, batch_dict, losses, pos_losses, neg_losses):
        # switch to train mode
        self.model.train()

        if torch.cuda.is_available():
            batch_dict = move_to_cuda(batch_dict)
        batch_size = batch_dict['batch_size']

        # compute output
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                outputs = self.model(**batch_dict)
        else:
            outputs = self.model(**batch_dict)

        scores = outputs['scores']
        pos_scores = scores[:batch_size] - self.args.additive_margin
        neg_scores = scores[batch_size:] + self.args.additive_margin
        neg_scores = torch.reshape(neg_scores, (batch_size, -1))

        if self.args.use_nas:
            #In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            neg_scores = (F.softmax(neg_scores * self.args.adversarial_temperature, dim = 1).detach() 
                                * F.logsigmoid(-neg_scores)).sum(dim = 1)
        else:
            neg_scores = F.logsigmoid(-neg_scores).mean(dim = 1)

        pos_scores = F.logsigmoid(pos_scores)

        if self.args.uni_weight:
            pos_loss = -pos_scores.mean()
            neg_loss = -neg_scores.mean()
        else:
            subsampling_weight = batch_dict['subsampling_weight']
            pos_loss = -(subsampling_weight * pos_scores).sum() / subsampling_weight.sum()
            neg_loss = -(subsampling_weight * neg_scores).sum() / subsampling_weight.sum()

        loss = (pos_loss + neg_loss) / 2

        losses.update(loss.item(), batch_size)
        pos_losses.update(pos_loss.item(), batch_size)
        neg_losses.update(neg_loss.item(), batch_size)

        # compute gradient and do SGD step
        self.optimizer.zero_grad()
        if self.args.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            self.optimizer.step()

        self.scheduler.step()

    def _setup_training(self):
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model).cuda()
        elif torch.cuda.is_available():
            self.model.cuda()
        else:
            logger.info('No gpu will be used')

    def _create_lr_scheduler(self, num_training_steps):
        if self.args.lr_scheduler == 'linear':
            return get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                   num_warmup_steps=self.args.warmup,
                                                   num_training_steps=num_training_steps)
        elif self.args.lr_scheduler == 'cosine':
            return get_cosine_schedule_with_warmup(optimizer=self.optimizer,
                                                   num_warmup_steps=self.args.warmup,
                                                   num_training_steps=num_training_steps)
        else:
            assert False, 'Unknown lr scheduler: {}'.format(self.args.scheduler)
