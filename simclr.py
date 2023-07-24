import logging
import os
import sys
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import save_config_file, accuracy, save_checkpoint
from aug_65to128 import aug_2nd
import numpy
import gc
#import objgraph
import time

torch.manual_seed(0)


class SimCLR(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        # args.device
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.batchsize = 2
        self.writer = SummaryWriter()
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        # CrossEntropyLoss
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)


    def info_nce_loss_1(self, features):
        labels = torch.zeros(64, 128)
        num = 1
        a = 0
        while (num <= 4):
            b = a + 8
            max = a + 7
            while (a <= max):
                labels[a, b] = 1
                labels[b, a] = 1
                a += 1
                b = a + 8
            a += 8
            num += 1

        labels1 = torch.zeros(64, 128)
        labels = torch.cat([labels, labels1], dim=0)
        labels = labels.to(self.args.device)

        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)


        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # positives
        positives = similarity_matrix[labels.bool()].view(64, -1)
        # negetives
        labels0 = labels.cpu()
        labels1 = torch.cat([labels0[0:64, :], torch.ones(64, 127)], 0)
        labels1 = labels1.to(self.args.device)

        negatives = similarity_matrix[~labels1.bool()].view(64, -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)
        logits = logits / self.args.temperature

        return logits, labels

    def info_nce_loss_2(self, features):
        labels = torch.zeros(128, 128)

        a = 64
        while (a <= 126):
            b = a + 1
            labels[a, b] = labels[b, a] = 1
            a += 2
        labels = labels.to(self.args.device)

        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)

        """feature"""
        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # positive
        positives = similarity_matrix[labels.bool()].view(64, -1)

        # negatives
        labels0 = labels.cpu()
        labels1 = torch.cat([torch.ones(64, 127), labels0[64:128, :]], 0)
        labels1 = labels1.to(self.args.device)
        negatives = similarity_matrix[~labels1.bool()].view(64, -1)


        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)
        logits = logits / self.args.temperature

        return logits, labels

    def train(self, dataloader):
        scaler = GradScaler(enabled=self.args.fp16_precision)

        save_config_file(self.writer.log_dir, self.args)

        n_iter = 100
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        for epoch in range(self.args.epochs):
            for images in tqdm(dataloader):

                #images1 = torch.cat(images1, dim=0)
                images = torch.squeeze(images)

                images = images.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    # features（2*batchsize，128）
                    # features = self.model(images)
                    features, feature_color = self.model(images)
                    logits, labels = self.info_nce_loss_1(features)
                    logits_c, labels_c = self.info_nce_loss_2(feature_color)
                    # loss
                    loss = 1 * self.criterion(logits, labels) + self.criterion(logits_c, labels_c)
                    loss1 = self.criterion(logits, labels)
                    loss_color = self.criterion(logits_c, labels_c)

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                if n_iter % self.args.log_every_n_steps == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    top1c, top5c = accuracy(logits_c, labels_c, topk=(1, 5))
                    self.writer.add_scalar('loss1', loss1, global_step=n_iter)
                    self.writer.add_scalar('loss_color', loss_color, global_step=n_iter)
                    self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                    self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                    self.writer.add_scalar('acc/top1_color', top1c[0], global_step=n_iter)
                    self.writer.add_scalar('acc/top5_color', top5c[0], global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)

                n_iter += 1

            if epoch >= 10:
                self.scheduler.step()
            logging.debug(
                f"Epoch: {epoch}\tLoss1: {loss1}\tTop1 accuracy: {top1[0]}\tTop5 accuracy:{top5[0]}\t"
                f"Loss_color: {loss_color}\tcolor_Top1 accuracy: {top1c[0]}\tcolor_Top5 accuracy:{top5c[0]}\t")


            if epoch == 20:
                checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(20)
                save_checkpoint({
                    'epoch': 20,
                    'arch': self.args.arch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
                logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")

            if epoch == 50:
                checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(50)
                save_checkpoint({
                    'epoch': 50,
                    'arch': self.args.arch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
                logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")

            if epoch == 100:
                checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(100)
                save_checkpoint({
                    'epoch': 100,
                    'arch': self.args.arch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
                logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")


        logging.info("Training has finished.")
        # save model checkpoints
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
        save_checkpoint({
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")



