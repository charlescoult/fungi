import argparse
import json
import os

from run import start_run
from metadata import RunMeta

def parse_args():
    parser = argparse.ArgumentParser(
        prog = 'Training Run',
        description = 'Input JSON run config file.',
    )
    parser.add_argument(
        'config_file',
    )
    parser.add_argument(
        '-t',
        '--test',
        action = 'store_true',
    )
    args = parser.parse_args()

    with open( args.config_file ) as f:
        run = json.load( f )

    return run, args.test

def make_dir_if_not_exist( runs_dir ):
    if ( os.path.exists( runs_dir ) ):
        if ( not os.path.isdir( runs_dir ) ):
            raise Exception( "File exists at runs_dir" )
    else:
        os.makedirs( runs_dir )


if __name__ == '__main__':

    run, test = parse_args()

    runs_dir = '/media/data/runs'
    runs_hdf = 'runs.h5'
    runs_hdf_key = 'runs'

    if ( test ):
        print( 'This is a test run.' )
        runs_dir = '/media/data/test_runs'
        print( 'Using %s' % runs_dir )

    make_dir_if_not_exist( runs_dir )

    run = RunMeta(
        run,
        runs_dir = runs_dir,
        runs_hdf = runs_hdf,
        runs_hdf_key = runs_hdf_key,
    )

    print(run)

    start_run( run )

