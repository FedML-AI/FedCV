import logging

import torch
from torch import nn

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, JsdCrossEntropy
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler


from FedML.fedml_core.trainer.model_trainer import ModelTrainer


class ClassificationTrainer(ModelTrainer):
    def __init__(self, model, device, args):
        super().__init__(model)
        # self.model = model
        self.args = args


        # self.lr_scheduler.step(0)

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
        # TODO
        # In fedavg, decay according to the round
        args.decay_epochs = args.decay_rounds
        if args.sched == 'step':
            self.lr_scheduler, self.num_epochs = create_scheduler(args, self.optimizer)
        elif args.sched == 'StepLR':
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 
                                                args.decay_epochs, args.decay_rate)
        elif args.sched is None:
            pass
        else:
            raise NotImplementedError

        # self.lr_scheduler.step(epoch=epoch + 1, metric=None)

        # This aims to make scheduler of torch works when scheduler is put before optimizer.
        # Please refer to pytorch document.
        if args.sched is not None:
            self.optimizer._step_count = 2
            self.lr_scheduler._step_count = 2
            self.lr_scheduler.step(epoch=args.round_idx)

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
                logging.info('Local Training Epoch: {} iter: {} \t Loss: {:.6f}'.format(
                                epoch, batch_idx, loss.item()))
            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                logging.info('(Trainer_ID {}. Local Training Epoch: {} \tLoss: {:.6f}'.format(
                    self.id, epoch, sum(epoch_loss) / len(epoch_loss)))



    def test(self, test_data, device, args):
        model = self.model

        model.eval()
        model.to(device)

        metrics = {
            'test_correct': 0,
            'test_loss': 0,
            'test_precision': 0,
            'test_recall': 0,
            'test_total': 0
        }

        # criterion = nn.CrossEntropyLoss().to(device)
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                loss = self.validate_loss_fn(pred, target)
                if args.dataset == "stackoverflow_lr":
                    predicted = (pred > .5).int()
                    correct = predicted.eq(target).sum(axis=-1).eq(target.size(1)).sum()
                    true_positive = ((target * predicted) > .1).int().sum(axis=-1)
                    precision = true_positive / (predicted.sum(axis=-1) + 1e-13)
                    recall = true_positive / (target.sum(axis=-1) + 1e-13)
                    metrics['test_precision'] += precision.sum().item()
                    metrics['test_recall'] += recall.sum().item()
                else:
                    _, predicted = torch.max(pred, -1)
                    correct = predicted.eq(target).sum()

                metrics['test_correct'] += correct.item()
                metrics['test_loss'] += loss.item() * target.size(0)
                metrics['test_total'] += target.size(0)
                logging.info('Local Testing iter: {} \t Loss: {:.6f} Acc: {:.6f}'.format(
                                batch_idx, loss.item(),  metrics['test_correct']/metrics['test_total']))
        return metrics

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        pass
