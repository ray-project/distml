from .input_pipeline import create_pretrain_dataset
import jax.numpy as jnp

import numpy as np
import numpy.random as npr

def make_wiki_train_loader(batch_size=8):
    inputs = ["./flax_util/sample_data_tfrecord/bert-pretrain-shard0.tfrecord"]

    train_dataset = create_pretrain_dataset(inputs, 
                                            seq_length=128, 
                                            max_predictions_per_seq=20, 
                                            batch_size=batch_size,
                                            is_training=True,
                                            use_next_sentence_label=True)
    return train_dataset


def tf2numpy(batch):
    new_batch = [dict()]
    for key in batch[0].keys():
        try:
            new_batch[0][key] = batch[0][key].numpy()
        except AttributeError:
            new_batch[0][key] = jnp.asarray(batch[0][key])
    try:
        new_batch.append(batch[1].numpy())
    except AttributeError:
        new_batch.append(jnp.asarray(batch[1]))
    del batch
    return new_batch


class Dataloader:
    def __init__(self, data, target, batch_size=128, shuffle=False):
        '''
        data: shape(width, height, channel, num)
        target: shape(num, num_classes)
        '''
        self.data = data
        self.target = target
        self.batch_size = batch_size
        num_data = self.target.shape[0]
        num_complete_batches, leftover = divmod(num_data, batch_size)
        self.num_batches = num_complete_batches + bool(leftover)
        self.shuffle = shuffle

    def synth_batches(self):
        num_imgs = self.target.shape[0]
        rng = npr.RandomState(npr.randint(10))
        perm = rng.permutation(num_imgs) if self.shuffle else np.arange(num_imgs)
        for i in range(self.num_batches):
            batch_idx = perm[i * self.batch_size:(i + 1) * self.batch_size]
            img_batch = self.data[batch_idx]
            label_batch = self.target[batch_idx]
            yield img_batch, label_batch

    def __iter__(self):
        return self.synth_batches()

    def __len__(self):
        return self.num_batches


if __name__ == "__main__":
    train_dataset = get_wiki_train_dataset()

    for batch in train_dataset:
        # print(batch)
        print(convert2numpy(batch))
        break
    # iterator = iter(train_dataset)
    # print(next(iterator))