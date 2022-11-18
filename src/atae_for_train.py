"""AttentionLSTM for training"""
from mindspore import nn
from mindspore import ops as P
from mindspore.common import dtype as mstype


class NetWithLoss(nn.Cell):
    """
    calculate loss
    """
    def __init__(self, model, batch_size=1):
        super(NetWithLoss, self).__init__()

        self.batch_size = batch_size
        self.model = model

        self.cast = P.Cast()
        self.transpose = P.Transpose()
        self.trans_matrix = (1, 0)
        self.cross_entropy = nn.BCELoss(reduction='sum')
        self.reduce_sum = P.ReduceSum()

    def construct(self, content, sen_len, aspect, solution):
        """
        content: (batch_size, 50) int32
        sen_len: (batch_size,) Int32
        aspect: (batch_size,) int32
        solution: (batch_size, 3) Int32
        """

        pred = self.model(content, sen_len, aspect)
        label = self.cast(solution, mstype.float32)

        loss = self.cross_entropy(pred, label)

        return loss
