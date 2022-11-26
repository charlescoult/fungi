import sys
import os
import io
import itertools

def main():
    fn = sys.argv[1]
    print( f'Fixing: {fn}' )

    new_string = ""

    with open( fn ) as f:
        in_tab_series = False
        for c in itertools.chain.from_iterable( f ):
            if ( not in_tab_series ):
                if (c == '\t'):
                    new_string += ','
                else:
                    new_string += c
            in_tab_series = (c == '\t')
                

    with open( fn + '.fixed', 'w' ) as f:
        f.write( new_string )

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nProgram, interrupted.')
        try:
            sys.exit(1)
        except SystemExit:
            os._exit(1)

