import os
import json
import time
import datetime
import logging
from util import limit_memory_growth

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import tensorflow_hub as hub

import numpy as np
import pandas as pd

import sklearn

version = '0.1.0'
__version__ = version

## Set logging to output INFO level to standard output
logging.basicConfig( level = os.environ.get( "LOGLEVEL", "INFO" ) )

## Set tf logging level to WARN
tf.get_logger().setLevel( 'ERROR' )

## tf autotune parameters
AUTOTUNE = tf.data.AUTOTUNE

limit_memory_growth()

## Multi-GPU strategy
# strategy = tf.distribute.MirroredStrategy( devices = [ "/gpu:0", "/gpu:1" ] )
strategy = tf.distribute.MirroredStrategy()

print("Imported common")
