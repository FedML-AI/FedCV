from copy import deepcopy


class MaxMeter(object):
    """
    Keeps track of the max of all the values that are 'add'ed
    """

    def __init__(self):
        self.max = None

    def update(self, value):
        """
        Add a value to the accumulator.
        :return: `true` if the provided value became the new max
        """
        if self.max is None or value > self.max:
            self.max = deepcopy(value)
            return True
        else:
            return False

    def value(self):
        """Access the current running average"""
        return self.max


class MinMeter(object):
    """
    Keeps track of the max of all the values that are 'add'ed
    """

    def __init__(self):
        self.min = None

    def update(self, value):
        """
        Add a value to the accumulator.
        :return: `true` if the provided value became the new max
        """
        if self.min is None or value < self.min:
            self.min = deepcopy(value)
            return True
        else:
            return False

    def value(self):
        """Access the current running average"""
        return self.min


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.max = -float("inf")
        self.min = float("inf")
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.max = val if val > self.max else self.max
        self.min = val if val < self.min else self.min


class RuntimeTracker(object):
    """Tracking the runtime stat for local training."""

    # def __init__(self, metrics_to_track=["top1"], on_cuda=True):
    def __init__(self, things_to_track=["loss"], on_cuda=True):
        self.things_to_track = things_to_track
        self.on_cuda = on_cuda
        self.n_samples = 0
        self.stat = None
        self.reset()


    def reset(self):
        self.stat = dict((name, AverageMeter()) for name in self.things_to_track)
        self.n_samples = 0

    # def evaluate_global_metric(self, metric):
    #     return global_average(
    #         self.stat[metric].sum, self.stat[metric].count, on_cuda=self.on_cuda
    #     ).item()

    # def evaluate_global_metrics(self):
    #     return [self.evaluate_global_metric(metric) for metric in self.metrics_to_track]

    def get_metrics_performance(self):
        return [self.stat[thing].avg for thing in self.things_to_track]

    def update_metrics(self, metric_stat, n_samples):
        self.n_samples += n_samples
        for thing in self.things_to_track:
            self.stat[thing].update(metric_stat[thing], n_samples)

    def __call__(self):
        return dict((name, val.avg) for name, val in self.stat.items())


class BestPerf(object):
    def __init__(self, best_perf=None, larger_is_better=True):
        self.best_perf = best_perf
        self.cur_perf = None
        self.best_perf_locs = []
        self.larger_is_better = larger_is_better

        # define meter
        self._define_meter()

    def _define_meter(self):
        self.meter = MaxMeter() if self.larger_is_better else MinMeter()

    def update(self, perf, perf_location):
        self.is_best = self.meter.update(perf)
        self.cur_perf = perf

        if self.is_best:
            self.best_perf = perf
            self.best_perf_locs += [perf_location]

    def get_best_perf_loc(self):
        return self.best_perf_locs[-1] if len(self.best_perf_locs) != 0 else None
