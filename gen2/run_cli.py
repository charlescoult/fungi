import argparse
import json

from run import start_run
from metadata import RunMeta

def parse_args():
    parser = argparse.ArgumentParser(
        description = 'Input JSON run config file.',
    )
    parser.add_argument( 'config_file' )
    args = parser.parse_args()

    with open( args.config_file ) as f:
        run = json.load( f )

    return run

if __name__ == '__main__':
    run = parse_args()

    run = RunMeta(
        run,
        runs_dir = '/media/data/runs',
        runs_hdf = 'runs.h5',
        runs_hdf_key = 'runs',
    )

    start_run( run )

