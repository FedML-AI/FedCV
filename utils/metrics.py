import math


class Metrics(object):

    def __init__(self, topks=[1], task="classification"):
        self.task = task
        self.topks = topks
        self.metric_names = self.get_metric_names(topks, task)
        self.metrics_fn = self._get_metric_measure(topks, task)

    def evaluate(self, loss, output, target):
        return self.metrics_fn(loss, output, target)

    @classmethod
    def get_metric_names(cls, topks, task):
        if task == "classification":
            metric_names = ["Acc{}".format(topk) for topk in topks]
            metric_names += ["Loss"]
        elif task == "stackoverflow_lr":
            metric_names = ["Acc", "Loss", "Precision", "Recall"]
        else:
            raise NotImplementedError
        return metric_names

    def _get_metric_measure(self, topks, task):
        if task == "classification":
            return self._classification_metric
        elif task == "stackoverflow_lr":
            return self._stackoverflow_lr_metric
        else:
            raise NotImplementedError
        
        assert self.metric_names is not None

    def _classification_metric(self, loss, output, target):
        """Computes the precision@k for the specified values of k"""
        metric_stat = {}
        metric_stat["Loss"] = loss.item()

        maxk = max(self.topks)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        for topk in self.topks:
            correct_k = correct[:topk].view(-1).float().sum(0, keepdim=True)
            # res.append(correct_k.mul_(100.0 / batch_size).item())
            metric_stat["Acc{}".format(topk)] = correct_k.mul_(100.0 / batch_size).item()

        return metric_stat

    def _stackoverflow_lr_metric(self, loss, output, target):
        metric_stat = {}
        metric_stat["Loss"] = loss.item()
        predicted = (output > .5).int()
        correct = predicted.eq(target).sum(axis=-1).eq(target.size(1)).sum()
        true_positive = ((target * predicted) > .1).int().sum(axis=-1)
        metric_stat["Precision"] = true_positive / (predicted.sum(axis=-1) + 1e-13)
        metric_stat["Recall"] = true_positive / (target.sum(axis=-1) + 1e-13)
        metric_stat["Acc"] = correct.mul_(100.0 / target.size(0)).item()
        metric_stat["Loss"] = loss.item()
        return metric_stat


