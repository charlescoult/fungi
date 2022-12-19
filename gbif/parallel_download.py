import requests
import os
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from urllib.parse import urlparse

import tensorflow as tf

def download_image( args ):
    print(args)

    # parse arguments
    url = args[0]
    if (len(args) == 1):
        fn = os.path.basename( urlparse(url).path )
        ext = ''
    else:
        fn = args[1]
        ext = os.path.splitext( urlparse(url).path )[1]

    # submit request and get response
    try:
        r = requests.get(url)
    except Exception as e:
        print('Exception in request:', e)

    image = r.content

    print( url )

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

    ## Convert back to raw
    image = tf.io.encode_jpeg( image ).numpy()

    # os.makedirs(os.path.dirname('media'), exist_ok=True)

    try:
        with open(os.path.join( 'media', str(fn) + ext ), 'wb') as f:
            f.write(image)
    except Exception as e:
        print('Exception in saving image:', e)

def download_parallel( args ):
    cpus = cpu_count()
    results = ThreadPool(cpus - 1).imap_unordered( download_image, args, chunksize=4 )
    for result in results:
        print( 'Result: ' + str(result) )

def download_sequential( args ):
    for arg in args:
        download_image( arg )
