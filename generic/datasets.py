import pathlib
import tensorflow as tf
import pandas as pd

flowers_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
flowers_data_dir = tf.keras.utils.get_file('flower_photos', origin=flowers_dataset_url, untar=True)
flowers_data_dir = pathlib.Path(flowers_data_dir)

class Dataset:

    datasets_meta = [
        ('CUB-200-2011', '/media/data/cub/CUB_200_2011/images'),
        ('flowers', flowers_data_dir),
        (
            'GBIF_fungi',
            '/media/data/gbif/media',
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


class CustomDataGen( tf.keras.utils.Sequence ):
    
    def __init__(
        self,
        df,
        X_col,
        y_col,
        batch_size,
        # input_size=(224, 224, 3),
        shuffle=True,
        augment=False,
    ):
        self.df = df.copy()
        self.X_col = X_col
        self.y_col = y_col
        self.batch_size = batch_size
        # self.input_size = input_size
        self.shuffle = shuffle
        
        self.n = len(self.df)
        self.n_name = df[y_col['name']].nunique()
        self.n_type = df[y_col['type']].nunique()
    
    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
    
    def __get_input(self, path, bbox, target_size):
    
        xmin, ymin, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']

        image = tf.keras.preprocessing.image.load_img(path)
        image_arr = tf.keras.preprocessing.image.img_to_array(image)

        image_arr = image_arr[ymin:ymin+h, xmin:xmin+w]
        image_arr = tf.image.resize(image_arr,(target_size[0], target_size[1])).numpy()

        return image_arr/255.
    
    def __get_output(self, label, num_classes):
        return tf.keras.utils.to_categorical(label, num_classes=num_classes)
    
    def __get_data(self, batches):
        # Generates data containing batch_size samples

        path_batch = batches[self.X_col['path']]
        bbox_batch = batches[self.X_col['bbox']]
        
        name_batch = batches[self.y_col['name']]
        type_batch = batches[self.y_col['type']]

        X_batch = np.asarray([self.__get_input(x, y, self.input_size) for x, y in zip(path_batch, bbox_batch)])

        y0_batch = np.asarray([self.__get_output(y, self.n_name) for y in name_batch])
        y1_batch = np.asarray([self.__get_output(y, self.n_type) for y in type_batch])

        return X_batch, tuple([y0_batch, y1_batch])
    
    def __getitem__(self, index):
        
        batches = self.df[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(batches)        
        return X, y
    
    def __len__(self):
        return self.n // self.batch_size



if __name__ == '__main__':

    hdf_file = '/media/data/clean_data.hd5'

    df = pd.read_hdf( hdf_file )

    print(df.columns)
    
    exit()
    datagen = CustomDataGen(

    )
