"""Dataset loader to feed into model."""
import mindspore.dataset as ds


def load_dataset(input_files, batch_size, sink_mode=False,
                 rank_size=1, rank_id=0):
    """
    Load dataset according to passed in params.

    Args:
        input_files (list): Data files.
        batch_size (int): Batch size.
        sink_mode (bool): Whether enable sink mode.
        rank_size (int): Rank size.
        rank_id (int): Rank id.
        shuffle (bool): Whether shuffle dataset.
        drop_remainder (bool): Whether drop the last possibly incomplete batch.
        is_translate (bool): Whether translate the text.

    Returns:
        Dataset, dataset instance.
    """
    if not input_files:
        raise ValueError("Require at least one dataset.")

    if not isinstance(sink_mode, bool):
        raise ValueError("`sink` must be type of bool.")

    data_set = ds.MindDataset(
        input_files, columns_list=["content", "sen_len", "aspect", "solution"],
        shuffle=False, num_shards=rank_size, shard_id=rank_id,
        num_parallel_workers=8)

    data_set = data_set.shuffle(buffer_size=data_set.get_dataset_size())

    ori_dataset_size = data_set.get_dataset_size()
    print(" | Dataset size", ori_dataset_size, ".")

    data_set = data_set.batch(batch_size=batch_size, drop_remainder=True)

    return data_set
