from common import *

class RunMeta(dict):

    def __init__(
        self,
        *args,
        runs_dir,
        runs_hdf,
        runs_hdf_key,
    ):
        super().__init__( *args )
        self.runs_dir = runs_dir
        self.runs_hdf = runs_hdf
        self.runs_hdf_key = runs_hdf_key

    # quick print of current run information for debug
    def print( self ):
        print( json.dumps( self.data , indent = 3 ) )

    # Will overwrite existing run in dataframe with the same id if one exists
    # - allows updating as we go
    def save(
        self,
        index = 'id',
    ):
        print( '\nSaving run metadata.\n' )
        # create df from run using json_normalize to flatten dict
        run_df = pd.json_normalize( self )
        run_df = run_df.set_index( index )

        # create runs_df if it doesn't exist
        runs_hdf_path = os.path.join( self.runs_dir, self.runs_hdf )
        if ( not os.path.isfile( runs_hdf_path ) ):
            pd.DataFrame().to_hdf( runs_hdf_path, self.runs_hdf_key )

        # read in the runs_hdf
        runs_df = pd.read_hdf(
            runs_hdf_path,
            self.runs_hdf_key,
        )

        # If a row for this run already exists, remove it
        if ( self[ index ] in runs_df.index ):
            runs_df = runs_df.drop( self[ index ] )

        # Add the updated data
        runs_df = pd.concat(
            [ runs_df, run_df ],
        )

        # save to file
        runs_df.to_hdf( runs_hdf_path, self.runs_hdf_key )

        # Make sure the run's path doesn't already exist and create it
        if ( not os.path.exists( self['path'] ) ):
            os.makedirs( self['path'] )

