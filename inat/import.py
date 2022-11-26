import pandas as pd
import pprint
import pyinaturalist as pin


my_loc = (35.59151610867911, -121.12048499955411)

obs = pin.get_observations( 
                       lat=my_loc[0],
                       lng=my_loc[1],
                       radius=10,
                       )


pin.pprint(obs)


