import os
import fire
import json
import numpy as np
import tensorflow as tf
import random
import time
import glob

from trainer.model import *
from trainer.sample import *
from trainer.encoder import *

CHECKPOINT_DIR = 'gs://dataseolabs/117M/checkpoint'
SAMPLE_DIR = 'gs://dataseolabs/117M/samples'
DOC_FILE = 'gs://dataseolabs/sherlock-holmes.txt'
PSEUDO = ''

def maketree(path):
    try:
        logging.warning(path)
        tf.gfile.MakeDirs(path)
        logging.warning("MAKETREE OK")
    except:
        logging.warning("MAKETREE ERROR")
        pass


def load_dataset(enc, path):
    paths = []

    # debug
    logging.warning(path)

    token_chunks = []

    with tf.gfile.GFile(path, "r") as f:
        logging.warning('TXT  ------ START')
        raw_text = f.read()
        logging.warning(len(raw_text))
        logging.warning('TXT  ------ END')

    tokens = np.stack(enc.encode(raw_text))
    token_chunks.append(tokens)

    return token_chunks


def binary_search(f, lo, hi):
    if f(lo) or not f(hi):
        return None
    while hi > lo + 1:
        mid = (lo + hi) // 2
        if f(mid):
            hi = mid
        else:
            lo = mid
    return hi


class Sampler(object):
    """Fairly samples a slice from a set of variable sized chunks.

    'Fairly' means that the distribution is the same as sampling from one concatenated chunk,
    but without crossing chunk boundaries."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.total_size = sum(chunk.shape[0] for chunk in chunks)
        self.boundaries = [0]
        for i in range(len(chunks)):
            self.boundaries.append(self.boundaries[-1] + chunks[i].shape[0])

    def sample(self, length):
        assert length < self.total_size // len(
            self.chunks
        ), "Dataset files are too small to sample {} tokens at a time".format(length)
        while True:
            index = random.randint(0, self.total_size - length - 1)
            i = binary_search(lambda j: self.boundaries[j] > index, 0,
                              len(self.boundaries) - 1) - 1
            if self.boundaries[i + 1] > index + length:
                within_chunk = index - self.boundaries[i]
                return self.chunks[i][within_chunk:within_chunk + length]

def train_main():

    dataset=DOC_FILE
    model_name='117M'
    seed=None
    batch_size=1
    sample_length=1023
    sample_num=1
    sample_every=50
    #sample_every=2
    # your pseudo
    run_name='run1'
    restore_from='latest'
    # 1000
    save_every=500
    #save_every=10

    enc = get_encoder(model_name)

    hparams = default_hparams()

    with tf.gfile.GFile("gs://dataseolabs/117M/hparams.json", "r") as f:
        logging.warning('HPARAMS ------ START')
        hparams.override_from_dict(json.load(f))
        logging.warning('HPARAMS ------ END')

    logging.warning(hparams)

    if sample_length is None:
        sample_length = hparams.n_ctx // 2
    elif sample_length > hparams.n_ctx:
        raise ValueError(
            "Can't get samples longer than window size: %s" % hparams.n_ctx)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)

        output = model(hparams=hparams, X=context)

        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=context[:, 1:], logits=output['logits'][:, :-1]))

        tf_sample = sample_sequence(
            hparams=hparams,
            length=sample_length,
            context=context,
            batch_size=batch_size,
            temperature=0.6,
            top_k=20)

        train_vars = [v for v in tf.trainable_variables() if 'model' in v.name]
        opt = tf.train.AdamOptimizer().minimize(loss, var_list=train_vars)

        saver = tf.train.Saver(
            var_list=train_vars,
            max_to_keep=5,
            # n_hours = 2
            keep_checkpoint_every_n_hours=1)
        sess.run(tf.global_variables_initializer())

        if restore_from == 'latest':
            logging.warning('restore lastest')
            ckpt = tf.train.latest_checkpoint(
                os.path.join(CHECKPOINT_DIR, run_name))

            if ckpt is None:
                # Get fresh GPT weights if new run.
                ckpt = tf.train.latest_checkpoint(
                    os.path.join('models', model_name))
                logging.warning('ckpt None -> OK')

        elif restore_from == 'fresh':
            logging.warning('restore fresh')
            ckpt = tf.train.latest_checkpoint(
                os.path.join('models', model_name))
        else:
            logging.warning('restore else')
            ckpt = tf.train.latest_checkpoint(restore_from)

        logging.warning('Loading checkpoint')
        logging.warning(ckpt)

        # find solution for restore
        # saver.restore(sess, ckpt)

        logging.warning('Loading dataset...')
        chunks = load_dataset(enc, dataset)
        data_sampler = Sampler(chunks)
        msg_data = 'dataset has '  + str(data_sampler.total_size) + ' tokens '
        logging.warning(msg_data)
        logging.warning('Training...')

        counter = 1

        logging.warning('find last counter...')
        logging.warning(os.path.join(CHECKPOINT_DIR, run_name, 'counter'))

        if tf.gfile.Exists(os.path.join(CHECKPOINT_DIR, run_name, 'counter')):
            # Load the step number if we're resuming a run
            # Add 1 so we don't immediately try to save again
            with tf.gfile.GFile(os.path.join(CHECKPOINT_DIR, run_name, 'counter'),
                      'r') as fp:
                counter = int(fp.read()) + 1

        logging.warning("counter = " + str(counter))

        def save():
            maketree(os.path.join(CHECKPOINT_DIR, run_name))
            logging.warning(
                'Saving',
                os.path.join(CHECKPOINT_DIR, run_name,
                             'model-{}').format(counter))
            saver.save(
                sess,
                os.path.join(CHECKPOINT_DIR, run_name, 'model'),
                global_step=counter)

            with tf.gfile.GFile(os.path.join(CHECKPOINT_DIR, run_name, 'counter'),'w') as fp:
                fp.write(str(counter) + '\n')

        def generate_samples():
            context_tokens = data_sampler.sample(1)
            all_text = []
            index = 0
            while index < sample_num:
                out = sess.run(
                    tf_sample, feed_dict={context: batch_size*[context_tokens]})
                for i in range(min(sample_num - index, batch_size)):
                    text = enc.decode(out[i])
                    text = '======== SAMPLE {} ========\n{}\n'.format(index + 1, text)
                    all_text.append(text)
                    index += 1
            logging.warning(text)

            # write samples
            maketree(os.path.join(SAMPLE_DIR, run_name))
            with tf.gfile.GFile(
                    os.path.join(SAMPLE_DIR, run_name,
                                 'samples-{}').format(counter), 'wb') as fp:
                # UnicodeEncodeError: 'ascii' codec can't encode character
                fp.write('\n'.join(all_text))

        avg_loss = (0.0, 0.0)
        start_time = time.time()

        try:
            while True:
                if counter % save_every == 0:
                    save()
                if counter % sample_every == 0:
                    generate_samples()

                batch = [data_sampler.sample(1024) for _ in range(batch_size)]

                _, lv = sess.run((opt, loss), feed_dict={context: batch})

                avg_loss = (avg_loss[0] * 0.99 + lv, avg_loss[1] * 0.99 + 1.0)

                logging.warning(
                    '[{counter} | {time:2.2f}] loss={loss:2.2f} avg={avg:2.2f}'
                    .format(
                        counter=counter,
                        time=time.time() - start_time,
                        loss=lv,
                        avg=avg_loss[0] / avg_loss[1]))

                counter += 1

        except KeyboardInterrupt:
            logging.warning('interrupted')
            save()


if __name__ == '__main__':
    train_main()
