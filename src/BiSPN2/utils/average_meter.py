# 在深度学习训练循环中，AverageMeter可以用于跟踪每个epoch或batch的损失、精度等指标。
# 通过不断更新AverageMeter实例，你可以轻松地获取这些指标的当前值和平均值，以便进行监控和调试。
class AverageMeter(object):
    """
    Computes and stores the average and current value of metrics.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        # 当前值
        self.val = 0
        # 平均值
        self.avg = 0
        # 总和
        self.sum = 0
        # 计数
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """
        String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)