import requests
import os
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from urllib.parse import urlparse

import tensorflow as tf

def download_image( args ):
    
    # parse arguments
    url = args[0]
    index = args[1]
    ext = args[2]
    fn = os.path.join('/mnt/gbif/media', str( index ) ) + ext

    # return if file already exists
    if (os.path.exists(fn)):
        print(f'File "{fn}" already exists')
        return

    # submit request and get response
    try:
        r = requests.get(url)
    except Exception as e:
        print('Exception in request:', e)
        return index
    
    image = r.content

    '''
    # resize image
    ## decode image to tf.image
    image = tf.io.decode_image( image )

    image = tf.image.resize(
            image,
            [ 256, 256 ],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
            preserve_aspect_ratio=True,
            # antialias=False,
            # name=None,
            )
    '''

    ## Convert back to raw
    # image = tf.io.encode_jpeg( image ).numpy()

    # os.makedirs(os.path.dirname('media'), exist_ok=True)
    
    try:
        with open( fn , 'wb' ) as f:
            f.write(image)
    except Exception as e:
        print('Exception in saving file:', e)
        return index

def download_parallel( args ):
    cpus = cpu_count()
    results = ThreadPool(cpus - 1).imap_unordered( download_image, args, chunksize=4 )
    errors = []
    for result in results:
        if (result):
            with open('errors.txt','a') as f:
                f.write(f'{result}\n')

def download_sequential( args ):
    for arg in args:
        download_image( arg )

def download( args, parallel = False ):
    if (parallel):
        download_parallel( args )
    else:
        download_sequential( args )