import tensorflow as tf

def limit_memory_growth(limit=True):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, limit)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)





import pandas as pd
import sklearn

def train_val_test_split(
    df,
    val_size = 0.1,
    test_size = 0.05,
):

    train_size = 1 - val_size - test_size
    if (train_size <= 0 ): raise Error("val_size + test_size >= 0")

    train, test = train_test_split(
        df,
        test_size = test_size,
        stratify = df[[ label_col ]],
        random_state = 42,
    )

    train, val = train_test_split(
        train,
        test_size = val_size,
        stratify= train[[ label_col ]],
        random_state = 42,
    )



import time

steps_per_epoch = 4

def timeit(
    ds,
    batch_size,
    batches = 2 * steps_per_epoch + 1,
):
    overall_start = time.time()
    # Fetch a single batch to prime the pipeline (fill the shuffle buffer),
    # before starting the timer
    it = iter(ds.take(batches+1))
    next(it)

    start = time.time()
    for i,(images,labels) in enumerate(it):
        if i%10 == 0:
            print('.',end='')
    print()
    end = time.time()

    duration = end-start
    print("{} batches: {} s".format(batches, duration))
    print("{:0.5f} Images/s".format(batch_size*batches/duration))
    print("Total time: {}s".format(end-overall_start))