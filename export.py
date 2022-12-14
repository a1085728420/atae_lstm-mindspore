"""export checkpoint file into air models"""

import numpy as np

from mindspore import Tensor, context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common import dtype as mstype
from mindspore.train.serialization import export
from mindspore import set_seed

from src.config import config
from src.model import AttentionLstm


context.set_context(
    mode=context.GRAPH_MODE,
    save_graphs=False,
    device_target="Ascend",
    device_id=0
)


if __name__ == "__main__":
    existed_ckpt = config.existed_ckpt

    word_path = config.word_path
    set_seed(config.rseed)
    r = np.load(word_path)
    word_vector = r['weight']
    weight = Tensor(word_vector, mstype.float32)

    network = AttentionLstm(config, weight, is_train=False)

    model_path = existed_ckpt
    ms_ckpt = load_checkpoint(model_path)
    load_param_into_net(network, ms_ckpt)

    x = Tensor(np.zeros((1, 50)).astype(np.int32))
    x_len = Tensor(np.array([20]).astype(np.int32))
    aspect = Tensor(np.array([1]).astype(np.int32))

    export(network, x, x_len, aspect, file_name=config.file_name, file_format=config.file_format)
    