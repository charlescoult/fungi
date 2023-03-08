import pathlib
import tensorflow as tf

flowers_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
flowers_data_dir = tf.keras.utils.get_file('flower_photos', origin=flowers_dataset_url, untar=True)
flowers_data_dir = pathlib.Path(flowers_data_dir)

class Dataset:

    datasets_meta = [
        ('CUB-200-2011', '/mnt/cub/CUB_200_2011/images'),
        ('flowers', flowers_data_dir),
        (
            'GBIF_fungi',
            '/mnt/gbif/media',
        ),
    ]


    def __init__(
        self,
        meta,
        image_size = 244,
        batch_size = 64,
        preprocess_input = None,
        validation_split = 0.05,
        seed = 42,
    ):
        self.meta = meta
        self.image_size = image_size
        self.batch_size = batch_size
        self.preprocess_input = preprocess_input
        self.validation_split = validation_split
        self.seed = seed
        self.build_dataset()

    # build a dataset object from a directory of images 
    def build_dataset(
        self,
    ):
        labels = 'infer'

        self.train, self.val = tf.keras.utils.image_dataset_from_directory(
            self.meta[1],
            batch_size = self.batch_size,
            validation_split = self.validation_split,
            image_size = (self.image_size, self.image_size),
            subset = "both",
            shuffle = True, # default but here for clarity
            seed = self.seed,
            label_mode = "categorical", # enables one-hot encoding (use 'int' for sparse_categorical_crossentropy loss)
            # labels = labels, # need to default to 'infer', not None
        )

        # Retrieve class names
        # (can't do this after converting to PrefetchDataset?)
        self.classes = self.train.class_names
        # Prefetch images
        # train = train.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        # val = val.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        self.train = self.train.prefetch(buffer_size=tf.data.AUTOTUNE)
        self.val = self.val.prefetch(buffer_size=tf.data.AUTOTUNE)

        # apply preprocessing function
        self.train = self.train.map(
            lambda x, y: (self.preprocess_input(x), y),
            num_parallel_calls = 16,
        )
        self.val = self.val.map(
            lambda x, y: (self.preprocess_input(x), y),
            num_parallel_calls = 16,
        )
