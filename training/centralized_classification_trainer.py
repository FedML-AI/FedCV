import logging

import torch
import torch.nn as nn

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, JsdCrossEntropy
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler

from FedML.fedml_core.trainer.model_trainer import ModelTrainer


class ClassificationTrainer(ModelTrainer):
    def __init__(self, model, device, args):
        super().__init__(model)
        # self.model = model
        self.args = args

        if args.opt in ['rmsproptf']:
            self.optimizer = create_optimizer(args, model)
        elif args.opt in ['momentum']:
             self.optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, 
                                            weight_decay=args.wd, momentum=args.momentum)
        elif args.opt in ['sgd']:
             self.optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, 
                                            weight_decay=args.wd)
        elif args.opt == 'adam':
            self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                         lr=args.lr,
                                         weight_decay=args.wd, amsgrad=True)
        else:
            raise NotImplementedError
        if args.sched == 'step':
            self.lr_scheduler, self.num_epochs = create_scheduler(args, self.optimizer)
        elif args.sched == 'StepLR':
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 
                                                args.decay_epochs, args.decay_rate)
        else:
            raise NotImplementedError
        self.lr_scheduler.step(0)

        # setup loss function
        # if args.jsd:
        #     assert num_aug_splits > 1  # JSD only valid with aug splits set
        #     self.train_loss_fn = JsdCrossEntropy(num_splits=num_aug_splits, smoothing=args.smoothing).to(device)
        # elif mixup_active:
        #     # smoothing is handled with mixup target transform
        #     self.train_loss_fn = SoftTargetCrossEntropy().to(device)
        if args.smoothing:
            self.train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing).to(device)
        else:
            self.train_loss_fn = nn.CrossEntropyLoss().to(device)
        self.validate_loss_fn = nn.CrossEntropyLoss().to(device)


    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args):
        model = self.model

        model.to(device)
        model.train()

        epoch_loss = []
        for epoch in range(args.epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                # logging.info(images.shape)
                x, labels = x.to(device), labels.to(device)
                self.optimizer.zero_grad()
                log_probs = model(x)
                loss = self.train_loss_fn(log_probs, labels)
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())
            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                logging.info('(Trainer_ID {}. Local Training Epoch: {} \tLoss: {:.6f}'.format(
                    self.id, epoch, sum(epoch_loss) / len(epoch_loss)))
            self.lr_scheduler.step(epoch=epoch + 1, metric=None)


    def train_one_epoch(self, train_data, device, args, epoch, tracker=None, metrics=None):
        model = self.model

        model.to(device)
        model.train()
        batch_loss = []
        for batch_idx, (x, labels) in enumerate(train_data):
            x, labels = x.to(device), labels.to(device)
            self.optimizer.zero_grad()
            log_probs = model(x)
            # logging.debug("labels: {}".format(labels))
            # logging.debug("pred: {}".format(log_probs))
            loss = self.train_loss_fn(log_probs, labels)
            loss.backward()
            self.optimizer.step()
            batch_loss.append(loss.item())
            if (metrics is not None) and (tracker is not None):
                metric_stat = metrics.evaluate(loss, log_probs, labels)
                tracker.update_metrics(metric_stat, n_samples=labels.size(0))
                if len(batch_loss) > 0:
                    logging.info('(Trainer_ID {}. Local Training Epoch: {}, Iter: {} \tLoss: {:.6f} ACC:{}'.format(
                        self.id, epoch, batch_idx, sum(batch_loss) / len(batch_loss), metric_stat['Acc']))
            else:
                if len(batch_loss) > 0:
                    logging.info('(Trainer_ID {}. Local Training Epoch: {}, Iter: {} \tLoss: {:.6f}'.format(
                        self.id, epoch, batch_idx, sum(batch_loss) / len(batch_loss)))
        self.lr_scheduler.step(epoch=epoch + 1, metric=None)

        if (metrics is not None) and (tracker is not None):
            return None
        else:
            return sum(batch_loss) / len(batch_loss)



    def train_one_step(self, train_batch_data, device, args, tracker=None, metrics=None):
        model = self.model

        model.to(device)
        model.train()
        x, labels = train_batch_data
        x, labels = x.to(device), labels.to(device)
        self.optimizer.zero_grad()
        log_probs = model(x)
        loss = self.train_loss_fn(log_probs, labels)
        loss.backward()
        self.optimizer.step()
        if (tracker is not None) and (metrics is not None): 
            metric_stat = metrics.evaluate(loss, log_probs, labels)
            tracker.update_metrics(metric_stat, n_samples=labels.size(0))

        return loss, log_probs, labels



    def test(self, test_data, device, args, tracker=None, metrics=None):
        model = self.model

        model.eval()
        model.to(device)


        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                # logging.debug("labels: {}".format(target))
                # logging.debug("pred: {}".format(pred))
                loss = self.validate_loss_fn(pred, target)
                if (metrics is not None) and (tracker is not None):
                    metric_stat = metrics.evaluate(loss, pred, target)
                    tracker.update_metrics(metric_stat, n_samples=target.size(0))
                    logging.info('(Trainer_ID {}. Local Testing Iter: {} \tLoss: {:.6f} ACC:{}'.format(
                        self.id, batch_idx, loss.item(), metric_stat['Acc']))
                else:
                    raise NotImplementedError

        if (metrics is not None) and (tracker is not None):
            return None
        else:
            raise NotImplementedError

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        pass




