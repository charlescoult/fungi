import tensorflow as tf
from sklearn.model_selection import train_test_split

# "For perfect shuffling, a buffer size greater than or equal to the full size of the dataset is required."
def train_val_test_split(
    ds,
    ds_size,
    train_split = None,
    val_split = 0.1,
    test_split = 0.1,
    shuffle_seed = None,
):

    if ( train_split == None ):
        train_split = 1 - val_split - test_split

    assert ( train_split + test_split + val_split ) == 1, "Split does not consume entire dataset"

    if shuffle_seed == None:
        shuffle_seed = tf.random.uniform(
            shape = [],
            dtype = tf.int64,
            minval = tf.int64.min,
            maxval = tf.int64.max,
        )

    # "reshuffle_each_iteration controls whether the shuffle order should be different for each epoch"
    # Specify seed to always have the same split distribution between runs
    ds = ds.shuffle(
        len(ds),
        seed = shuffle_seed,
        name = 'train_val_test_split_shuffle',
    )

    train_size = int( train_split * len(ds) )
    val_size = int( val_split * len(ds) )

    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)

    # return shuffle_seed for future repeatability, if user desires
    return train_ds, val_ds, test_ds, shuffle_seed

def train_val_test_split_stratified(
    ds,
    col_label,
    train_split = None,
    val_split = 0.1,
    test_split = 0.1,
    shuffle_seed = None,
):

    val_split_2 = val_split / ( 1 - test_split )

    if ( train_split == None ):
        train_split = 1 - val_split - test_split

    assert ( train_split + test_split + val_split ) == 1, "Split does not consume entire dataset"

    if shuffle_seed == None:
        shuffle_seed = tf.random.uniform(
            shape = [],
            dtype = tf.int64,
            minval = 0,
            maxval = 2**32 - 1,
        ).numpy()

    # test
    ds_train, ds_test = train_test_split(
        ds,
        test_size = test_split,
        stratify = ds[col_label],
        random_state = shuffle_seed,
    )

    # val
    ds_train, ds_val = train_test_split(
        ds_train,
        test_size = val_split_2,
        stratify = ds_train[col_label],
        random_state = shuffle_seed,
    )

    return ds_train, ds_val, ds_test, shuffle_seed


