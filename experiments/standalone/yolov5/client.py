import logging

import os
import torch
from torch import nn


import time
from pathlib import Path
from threading import Thread
from warnings import warn

import math
import random
import numpy as np
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
# from apex import amp
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import test  # import test.py to get mAP after each epoch
# from fedml_api.model.object_detection.yolov5.models.experimental import
from fedml_api.model.object_detection.yolov5.models.experimental import attempt_load
from fedml_api.model.object_detection.yolov5.models.yolo import Model
from fedml_api.model.object_detection.yolov5.utils.autoanchor import check_anchors
# from utils.datasets import create_dataloader
from fedml_api.model.object_detection.yolov5.utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    fitness, strip_optimizer, get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    print_mutation, set_logging
from fedml_api.model.object_detection.yolov5.utils.google_utils import attempt_download
from fedml_api.model.object_detection.yolov5.utils.loss import compute_loss
from fedml_api.model.object_detection.yolov5.utils.plots import plot_images, plot_labels, plot_results, plot_evolution
from fedml_api.model.object_detection.yolov5.utils.torch_utils import ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first

from fedml_api.data_preprocessing.coco_detection.datasets import partition_data
from fedml_api.data_preprocessing.coco_detection.datasets import create_dataloader



logger = logging.getLogger(__name__)


class Client:

    def __init__(self, client_idx, local_training_data, local_sample_number, opt, device, model, tb_writer, wandb, hyp):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        # self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        logging.info("self.local_sample_number = " + str(self.local_sample_number))

        self.opt = opt
        self.device = device
        self.model = model
        self.hyp = hyp
        self.tb_writer = tb_writer
        self.wandb = wandb

        '''
        stackoverflow_lr is the task of multi-label classification
        please refer to following links for detailed explainations on cross-entropy and corresponding implementation of tff research:
        https://towardsdatascience.com/cross-entropy-for-classification-d98e7f974451
        https://github.com/google-research/federated/blob/49a43456aa5eaee3e1749855eed89c0087983541/optimization/stackoverflow_lr/federated_stackoverflow_lr.py#L131
        '''
        # if self.args.dataset == "stackoverflow_lr":
        #     self.criterion = nn.BCELoss(reduction = 'sum').to(device)
        # else:
        #     self.criterion = nn.CrossEntropyLoss().to(device)

    def update_local_dataset(self, client_idx, local_training_data, local_sample_number):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        # self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number

    def get_sample_number(self):
        return self.local_sample_number

    def train(self, w_global, dataset, dataloader, wandb):
        self.wandb = wandb
        logger.info(f'Hyperparameters {self.hyp}')
        save_dir, epochs, batch_size, total_batch_size, weights, rank = \
            Path(
                self.opt.save_dir), self.opt.epochs, self.opt.batch_size, self.opt.total_batch_size, self.opt.weights, self.opt.global_rank

        # Directories
        wdir = save_dir / 'weights'
        wdir.mkdir(parents=True, exist_ok=True)  # make dir
        last = wdir / 'last.pt'
        best = wdir / 'best.pt'
        results_file = save_dir / 'results.txt'

        # Save run settings
        with open(save_dir / 'hyp.yaml', 'w') as f:
            yaml.dump(self.hyp, f, sort_keys=False)
        with open(save_dir / 'opt.yaml', 'w') as f:
            yaml.dump(vars(self.opt), f, sort_keys=False)

        # Configure
        plots = not self.opt.evolve  # create plots
        cuda = self.device.type != 'cpu'
        init_seeds(2 + rank)
        with open(self.opt.data) as f:
            data_dict = yaml.load(f, Loader=yaml.FullLoader)  # data dict
        with torch_distributed_zero_first(rank):
            check_dataset(data_dict)  # check
        train_path = data_dict['train']
        test_path = data_dict['val']
        nc, names = (1, ['item']) if self.opt.single_cls else (
            int(data_dict['nc']), data_dict['names'])  # number classes, names
        assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check

        # Model
        pretrained = weights.endswith('.pt')
        if pretrained:
            with torch_distributed_zero_first(rank):
                attempt_download(weights)  # download if not found locally
            ckpt = torch.load(weights, map_location=self.device)  # load checkpoint
            if self.hyp.get('anchors'):
                ckpt['model'].yaml['anchors'] = round(self.hyp['anchors'])  # force autoanchor
            model = Model(self.opt.cfg or ckpt['model'].yaml, ch=3, nc=nc).to(self.device)  # create
            exclude = ['anchor'] if self.opt.cfg or self.hyp.get('anchors') else []  # exclude keys
            state_dict = ckpt['model'].float().state_dict()  # to FP32
            state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
            model.load_state_dict(state_dict, strict=False)  # load
            logger.info(
                'Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weights))  # report
        else:
            model = Model(self.opt.cfg, ch=3, nc=nc).to(self.device)  # create

        # Freeze
        freeze = []  # parameter names to freeze (full or partial)
        for k, v in model.named_parameters():
            v.requires_grad = True  # train all layers
            if any(x in k for x in freeze):
                print('freezing %s' % k)
                v.requires_grad = False

        # Optimizer
        nbs = 64  # nominal batch size
        accumulate = max(round(nbs / total_batch_size), 1)  # accumulate loss before optimizing
        self.hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # scale weight_decay

        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
        for k, v in model.named_modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)  # biases
            if isinstance(v, nn.BatchNorm2d):
                pg0.append(v.weight)  # no decay
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)  # apply decay

        if self.opt.adam:
            optimizer = optim.Adam(pg0, lr=self.hyp['lr0'], betas=(self.hyp['momentum'], 0.999))  # adjust beta1 to momentum
        else:
            optimizer = optim.SGD(pg0, lr=self.hyp['lr0'], momentum=self.hyp['momentum'], nesterov=True)

        optimizer.add_param_group({'params': pg1, 'weight_decay': self.hyp['weight_decay']})  # add pg1 with weight_decay
        optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
        logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
        del pg0, pg1, pg2

        # Scheduler https://arxiv.org/pdf/1812.01187.pdf
        # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
        lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - self.hyp['lrf']) + self.hyp['lrf']  # cosine
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
        # plot_lr_scheduler(optimizer, scheduler, epochs)

        # Logging
        if self.wandb and self.wandb.run is None:
            self.opt.hyp = self.hyp  # add hyperparameters
            wandb_run = self.wandb.init(config=self.opt, resume="allow",
                                        project='YOLOv5' if self.opt.project == 'runs/train' else Path(
                                            self.opt.project).stem,
                                        name=save_dir.stem,
                                        id=ckpt.get('wandb_id') if 'ckpt' in locals() else None)
        loggers = {'wandb': self.wandb}  # loggers dict

        # Resume
        start_epoch, best_fitness = 0, 0.0
        if pretrained:
            # Optimizer
            if ckpt['optimizer'] is not None:
                optimizer.load_state_dict(ckpt['optimizer'])
                best_fitness = ckpt['best_fitness']

            # Results
            if ckpt.get('training_results') is not None:
                with open(results_file, 'w') as file:
                    file.write(ckpt['training_results'])  # write results.txt

            # Epochs
            start_epoch = ckpt['epoch'] + 1
            if self.opt.resume:
                assert start_epoch > 0, '%s training to %g epochs is finished, nothing to resume.' % (weights, epochs)
            if epochs < start_epoch:
                logger.info('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                            (weights, ckpt['epoch'], epochs))
                epochs += ckpt['epoch']  # finetune additional epochs

            del ckpt, state_dict

        # Image sizes
        gs = int(max(model.stride))  # grid size (max stride)
        imgsz, imgsz_test = [check_img_size(x, gs) for x in self.opt.img_size]  # verify imgsz are gs-multiples

        # DP mode
        if cuda and rank == -1 and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)

        # SyncBatchNorm
        if self.opt.sync_bn and cuda and rank != -1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(self.device)
            logger.info('Using SyncBatchNorm()')

        # EMA
        ema = ModelEMA(model) if rank in [-1, 0] else None

        # DDP mode
        if cuda and rank != -1:
            model = DDP(model, device_ids=[self.opt.local_rank], output_device=self.opt.local_rank)

        # Trainloader
        # dataloader, dataset = create_dataloader(train_path, imgsz, batch_size, gs, self.opt,
        #                                         hyp=self.hyp, augment=True, cache=self.opt.cache_images, rect=self.opt.rect,
        #                                         rank=rank,
        #                                         world_size=self.opt.world_size, workers=self.opt.workers,
        #                                         image_weights=self.opt.image_weights)

        # client
        # client_number = self.opt.client_number
        # partition = self.opt.partition
        # net_dataidx_map = partition_data(train_path, partition=partition, n_nets=client_number)
        # train_data_loader_dict = dict()
        # for i in range(client_number):
        #     dataloader, dataset = create_dataloader(train_path, imgsz, batch_size, gs, self.opt,
        #                                             hyp=self.hyp, augment=True, cache=self.opt.cache_images,
        #                                             rect=self.opt.rect,
        #                                             rank=rank,
        #                                             world_size=self.opt.world_size, workers=self.opt.workers,
        #                                             image_weights=self.opt.image_weights,
        #                                             net_dataidx_map=net_dataidx_map[i])
        #
        #     train_data_loader_dict[i] = dataloader
            # self.client_list.append(Client(i, train_data_loader_dict[i], len(dataset), self.opt, self.device, model))

        # TODO: train_client
        # client sampling
        # client train
        # logging info

        # train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])
        # client_number_per_round = self.opt.client_num_per_round

        mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
        nb = len(dataloader)  # number of batches
        print("nb:", nb)
        assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (
            mlc, nc, self.opt.data, nc - 1)

        # Process 0
        if rank in [-1, 0]:
            ema.updates = start_epoch * nb // accumulate  # set EMA updates
            testloader = create_dataloader(test_path, imgsz_test, total_batch_size, gs, self.opt,  # testloader
                                           hyp=self.hyp, cache=self.opt.cache_images and not self.opt.notest, rect=True,
                                           rank=-1, world_size=self.opt.world_size, workers=self.opt.workers, pad=0.5)[
                0]

            if not self.opt.resume:
                labels = np.concatenate(dataset.labels, 0)
                c = torch.tensor(labels[:, 0])  # classes
                # cf = torch.bincount(c.long(), minlength=nc) + 1.  # frequency
                # model._initialize_biases(cf.to(self.device))
                if plots:
                    Thread(target=plot_labels, args=(labels, save_dir, loggers), daemon=True).start()
                    if self.tb_writer:
                        self.tb_writer.add_histogram('classes', c, 0)

                # Anchors
                if not self.opt.noautoanchor:
                    check_anchors(dataset, model=model, thr=self.hyp['anchor_t'], imgsz=imgsz)

        # Model parameters
        self.hyp['cls'] *= nc / 80.  # scale coco-tuned hyp['cls'] to current dataset
        model.nc = nc  # attach number of classes to model
        model.hyp = self.hyp  # attach hyperparameters to model
        model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
        model.class_weights = labels_to_class_weights(dataset.labels, nc).to(self.device)  # attach class weights
        model.names = names

        # Start training
        t0 = time.time()
        nw = max(round(self.hyp['warmup_epochs'] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
        # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
        maps = np.zeros(nc)  # mAP per class
        results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
        scheduler.last_epoch = start_epoch - 1  # do not move
        scaler = amp.GradScaler(enabled=cuda)
        logger.info('Image sizes %g train, %g test\n'
                    'Using %g dataloader workers\nLogging results to %s\n'
                    'Starting training for %g epochs...' % (
                        imgsz, imgsz_test, dataloader.num_workers, save_dir, epochs))
        model = self.model
        model.load_state_dict(w_global)
        model.to(self.device)
        for epoch in range(start_epoch,
                           epochs):  # epoch ------------------------------------------------------------------
            model.train()

            # client_indexes = client_sampling(epoch, client_number, client_number_per_round)
            # logging.info("client_indexes = " + str(client_indexes))

            # Update image weights (optional)
            if self.opt.image_weights:
                # Generate indices
                if rank in [-1, 0]:
                    cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2  # class weights
                    iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
                    dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx
                # Broadcast if DDP
                if rank != -1:
                    indices = (torch.tensor(dataset.indices) if rank == 0 else torch.zeros(dataset.n)).int()
                    dist.broadcast(indices, 0)
                    if rank != 0:
                        dataset.indices = indices.cpu().numpy()

            # Update mosaic border
            # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
            # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

            mloss = torch.zeros(4, device=self.device)  # mean losses
            if rank != -1:
                dataloader.sampler.set_epoch(epoch)
            pbar = enumerate(dataloader)
            logger.info(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'total', 'targets', 'img_size'))
            if rank in [-1, 0]:
                pbar = tqdm(pbar, total=nb)  # progress bar
            optimizer.zero_grad()
            for i, (
                    imgs, targets, paths,
                    _) in pbar:  # batch -------------------------------------------------------------
                ni = i + nb * epoch  # number integrated batches (since train start)
                imgs = imgs.to(self.device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0

                # Warmup
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                    accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())
                    for j, x in enumerate(optimizer.param_groups):
                        # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x['lr'] = np.interp(ni, xi,
                                            [self.hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                        if 'momentum' in x:
                            x['momentum'] = np.interp(ni, xi, [self.hyp['warmup_momentum'],self.hyp['momentum']])

                # Multi-scale
                if self.opt.multi_scale:
                    sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                    sf = sz / max(imgs.shape[2:])  # scale factor
                    if sf != 1:
                        ns = [math.ceil(x * sf / gs) * gs for x in
                              imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                        imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

                # Forward
                with amp.autocast(enabled=cuda):
                    pred = model(imgs)  # forward
                    loss, loss_items = compute_loss(pred, targets.to(self.device), model)  # loss scaled by batch_size
                    if rank != -1:
                        loss *= self.opt.world_size  # gradient averaged between devices in DDP mode

                # Backward
                scaler.scale(loss).backward()

                # Optimize
                # if ni % accumulate == 0:
                #     scaler.step(optimizer)  # optimizer.step
                #     scaler.update()
                #     optimizer.zero_grad()
                #     if ema:
                #         ema.update(model)

                # Print
                if rank in [-1, 0]:
                    mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                    mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                    s = ('%10s' * 2 + '%10.4g' * 6) % (
                        '%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
                    pbar.set_description(s)

                    # Plot
                    if plots and ni < 3:
                        f = save_dir / f'train_batch{ni}.jpg'  # filename
                        Thread(target=plot_images, args=(imgs, targets, paths, f), daemon=True).start()
                        # if tb_writer:
                        #     tb_writer.add_image(f, result, dataformats='HWC', global_step=epoch)
                        #     tb_writer.add_graph(model, imgs)  # add model to tensorboard
                    elif plots and ni == 3 and self.wandb:
                        self.wandb.log(
                            {"Mosaics": [self.wandb.Image(str(x), caption=x.name) for x in
                                         save_dir.glob('train*.jpg')]})

                # end batch ------------------------------------------------------------------------------------------------
            # end epoch ----------------------------------------------------------------------------------------------------

            # Scheduler
            lr = [x['lr'] for x in optimizer.param_groups]  # for tensorboard
            scheduler.step()

            # DDP process 0 or single-GPU
            if rank in [-1, 0]:
                # mAP
                if ema:
                    ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride'])
                final_epoch = epoch + 1 == epochs
                if not self.opt.notest or final_epoch:  # Calculate mAP
                    results, maps, times = test.test(self.opt.data,
                                                     batch_size=total_batch_size,
                                                     imgsz=imgsz_test,
                                                     model=ema.ema,
                                                     single_cls=self.opt.single_cls,
                                                     dataloader=testloader,
                                                     save_dir=save_dir,
                                                     plots=plots and final_epoch,
                                                     log_imgs=self.opt.log_imgs if self.wandb else 0)

                # Write
                with open(results_file, 'a') as f:
                    f.write(s + '%10.4g' * 7 % results + '\n')  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
                if len(self.opt.name) and self.opt.bucket:
                    os.system(
                        'gsutil cp %s gs://%s/results/results%s.txt' % (results_file, self.opt.bucket, self.opt.name))

                # Log
                tags = ['train/box_loss', 'train/obj_loss', 'train/cls_loss',  # train loss
                        'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
                        'val/box_loss', 'val/obj_loss', 'val/cls_loss',  # val loss
                        'x/lr0', 'x/lr1', 'x/lr2']  # params
                for x, tag in zip(list(mloss[:-1]) + list(results) + lr, tags):
                    if self.tb_writer:
                        self.tb_writer.add_scalar(tag, x, epoch)  # tensorboard
                    if self.wandb:
                        self.wandb.log({tag: x})  # W&B

                # Update best mAP
                fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
                if fi > best_fitness:
                    best_fitness = fi

                # Save model
                save = (not self.opt.nosave) or (final_epoch and not self.opt.evolve)
                if save:
                    with open(results_file, 'r') as f:  # create checkpoint
                        ckpt = {'epoch': epoch,
                                'best_fitness': best_fitness,
                                'training_results': f.read(),
                                'model': ema.ema,
                                'optimizer': None if final_epoch else optimizer.state_dict(),
                                'wandb_id': wandb_run.id if self.wandb else None}

                    # Save last, best and delete
                    torch.save(ckpt, last)
                    if best_fitness == fi:
                        torch.save(ckpt, best)
                    del ckpt
            # end epoch ----------------------------------------------------------------------------------------------------
        # end training

        if rank in [-1, 0]:
            # Strip optimizers
            for f in [last, best]:
                if f.exists():  # is *.pt
                    strip_optimizer(f)  # strip optimizer
                    os.system(
                        'gsutil cp %s gs://%s/weights' % (f, self.opt.bucket)) if self.opt.bucket else None  # upload

            # Plots
            if plots:
                plot_results(save_dir=save_dir)  # save as results.png
                if self.wandb:
                    files = ['results.png', 'precision_recall_curve.png', 'confusion_matrix.png']
                    self.wandb.log({"Results": [self.wandb.Image(str(save_dir / f), caption=f) for f in files
                                                if (save_dir / f).exists()]})
            logger.info('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))

            # Test best.pt
            # if self.opt.data.endswith('coco.yaml') and nc == 80:  # if COCO
            #     results, _, _ = test.test(self.opt.data,
            #                               batch_size=total_batch_size,
            #                               imgsz=imgsz_test,
            #                               model=attempt_load(best if best.exists() else last, self.device).half(),
            #                               single_cls=self.opt.single_cls,
            #                               dataloader=testloader,
            #                               save_dir=save_dir,
            #                               save_json=True,  # use pycocotools
            #                               plots=False)

        else:
            dist.destroy_process_group()

        self.wandb.run.finish() if self.wandb and self.wandb.run else None
        torch.cuda.empty_cache()
        return model.cpu().state_dict(), mloss #, results

    # def train(self, w_global):
    #     self.model.train()
    #     self.model.load_state_dict(w_global)
    #     self.model.to(self.device)
    #
    #     # train and update
    #     if self.args.client_optimizer == "sgd":
    #         optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr)
    #     else:
    #         optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.lr,
    #                                           weight_decay=self.args.wd, amsgrad=True)
    #
    #     epoch_loss = []
    #     for epoch in range(self.args.epochs):
    #         batch_loss = []
    #         for batch_idx, (x, labels) in enumerate(self.local_training_data):
    #             x, labels = x.to(self.device), labels.to(self.device)
    #             # logging.info("x.size = " + str(x.size()))
    #             # logging.info("labels.size = " + str(labels.size()))
    #             self.model.zero_grad()
    #             log_probs = self.model(x)
    #             loss = self.criterion(log_probs, labels)
    #             loss.backward()
    #
    #             # to avoid nan loss
    #             # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
    #
    #             optimizer.step()
    #             # logging.info('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
    #             #     epoch, (batch_idx + 1) * self.args.batch_size, len(self.local_training_data) * self.args.batch_size,
    #             #            100. * (batch_idx + 1) / len(self.local_training_data), loss.item()))
    #             batch_loss.append(loss.item())
    #         epoch_loss.append(sum(batch_loss) / len(batch_loss))
    #         # logging.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(
    #         #     self.client_idx, epoch, sum(epoch_loss) / len(epoch_loss)))
    #     return self.model.cpu().state_dict(), sum(epoch_loss) / len(epoch_loss)

    def local_test(self, model_global, b_use_test_dataset=False):
        model_global.eval()
        model_global.to(self.device)
        metrics = {
            'test_correct': 0,
            'test_loss' : 0,
            'test_precision': 0,
            'test_recall': 0,
            'test_total' : 0
        }
        if b_use_test_dataset:
            test_data = self.local_test_data
        else:
            test_data = self.local_training_data
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(self.device)
                target = target.to(self.device)
                pred = model_global(x)
                loss = self.criterion(pred, target)

                if self.args.dataset == "stackoverflow_lr":
                    predicted = (pred > .5).int()
                    correct = predicted.eq(target).sum(axis = -1).eq(target.size(1)).sum()
                    true_positive = ((target * predicted) > .1).int().sum(axis = -1)
                    precision = true_positive / (predicted.sum(axis = -1) + 1e-13)
                    recall = true_positive / (target.sum(axis = -1)  + 1e-13)
                    metrics['test_precision'] += precision.sum().item()
                    metrics['test_recall'] += recall.sum().item()
                else:
                    _, predicted = torch.max(pred, 1)
                    correct = predicted.eq(target).sum()

                metrics['test_correct'] += correct.item()
                metrics['test_loss'] += loss.item() * target.size(0)
                if len(target.size()) == 1: #
                    metrics['test_total'] += target.size(0)
                elif len(target.size()) == 2: # for tasks of next word prediction
                    metrics['test_total'] += target.size(0) * target.size(1)

        return metrics
