import requests
import os
import time
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from urllib.parse import urlparse

def download_image( args ):
    print(args)
    t0 = time.time()
    url = args[0]
    if (len(args) == 1):
        fn = os.path.basename( urlparse(url).path )
        ext = ''
    else:
        fn = args[1]
        ext = os.path.splitext( urlparse(url).path )[1]

    try:
        r = requests.get(url)
    except Exception as e:
        print('Exception in request:', e)

    try:
        with open(os.path.join( 'media', str(fn) + ext ), 'wb') as f:
            f.write(r.content)
        return(url, time.time() - t0)
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
