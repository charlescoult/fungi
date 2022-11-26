
from tqdm import tqdm

import json
from datetime import datetime
from dateutil import tz
from os.path import exists
from pprint import pprint

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import dates
from matplotlib import pyplot as plt

from pyinaturalist import get_observations, get_places_autocomplete, get_observation_histogram

BASIC_OBS_COLUMNS = [
        'id', 'observed_on', 'location', 'uri', 'taxon.id',
        'taxon.name', 'taxon.rank', 'taxon.preferred_common_name', 'user.login',
        ]
DATASET_FILENAME = 'midwest_monarchs.json'
PLOT_COLOR = '#fa7b23'
MIDWEST_STATE_IDS = [3, 20, 24, 25, 28, 32, 35, 38]  # place_ids of 8 states in the Midwest US

sns.set_theme(style="darkgrid")

def date_to_mpl_day_of_year(dt):
    """Get a matplotlib-compatible date number, ignoring the year (to represent day of year)"""
    try:
        return dates.date2num(dt.replace(year=datetime.now().year))
    except ValueError:
        return None

def date_to_mpl_time(dt):
    """Get a matplotlib-compatible date number, ignoring the date (to represent time of day)"""
    try:
        return date_to_num(dt) % 1
    except ValueError:
        return None

def to_local_tz(dt):
    """Convert a datetime object to the local time zone"""
    try:
        return dt.astimezone(tz.tzlocal())
    except (TypeError, ValueError):
        return None

def get_xlim():
    """Get limits of x axis for first and last days of the year"""
    now = datetime.now()
    xmin = dates.date2num(datetime(now.year, 1, 1))
    xmax = dates.date2num(datetime(now.year, 12, 31))
    return xmin, xmax

def get_colormap(color):
    """Make a colormap (gradient) based on the given color; copied from seaborn.axisgrid"""
    color_rgb = mpl.colors.colorConverter.to_rgb(color)
    colors = [sns.set_hls_values(color_rgb, l=l) for l in np.linspace(1, 0, 12)]
    return sns.blend_palette(colors, as_cmap=True)

def pdir(obj, sort_types=False, non_callables=False):
    attrs = {attr: type(getattr(obj, attr)).__name__ for attr in dir(obj)}
    if sort_types:
        attrs = {k: v for k, v in sorted(attrs.items(), key=lambda x: x[1])}
    if non_callables:
        attrs = {k: v for k, v in attrs.items() if v not in ['function', 'method', 'method-wrapper', 'builtin_function_or_method']}
    pprint(attrs, sort_dicts=not sort_types)

'''
print()
print("GETTING OBS")
observations = get_observations(
    taxon_name='Danaus plexippus',
    photos=True,
    geo=True,
    geoprivacy='open',
    place_id=MIDWEST_STATE_IDS,
    # page='all',
)

total_pages = int( observations['total_results'] / observations['per_page'] ) + 1
all_observations = []
all_observations.append( observations['results'] )

for page in tqdm(range( total_pages )):
    observations = get_observations(
            taxon_name='Danaus plexippus',
            photos=True,
            geo=True,
            geoprivacy='open',
            place_id=MIDWEST_STATE_IDS,
            page = page,
        )
    all_observations.append( observations['results'] )

print("DONE")

pprint(observations['total_results'])
pprint(observations['per_page'])
pprint(len(observations['results']) )
print()

# Save results for future usage
with open(DATASET_FILENAME, 'w') as f:
    json.dump(observations, f, indent=4, sort_keys=True, default=str)

print(f'Total observations: {len(observations)}')

# Flatten nested JSON values
df = pd.json_normalize(observations)

print( df.columns )

# Normalize timezones
df['observed_on'] = df['observed_on'].dropna().apply(to_local_tz)

# Add some extra date/time columns that matplotlib can more easily handle
df['observed_time_mp'] = df['observed_on'].apply(date_to_mpl_time)
df['observed_on_mp'] = df['observed_on'].apply(date_to_mpl_day_of_year)



# Preview: Show counts by month observed X quality grade
df['observed_month'] = df['observed_on'].apply(lambda x: x.month)
df[['observed_month', 'quality_grade']].groupby(['observed_month', 'quality_grade']).size().reset_index(name='counts')
'''


# Use histogram endpoint instead

year_hist = get_observation_histogram(
            taxon_name='Danaus plexippus',
            photos=True,
            # geo=True,
            # geoprivacy='open',
            interval='year',
            # date_field='created',
            place_id=MIDWEST_STATE_IDS,
        )

month_hist = get_observation_histogram(
            taxon_name='Danaus plexippus',
            photos=True,
            # geo=True,
            # geoprivacy='open',
            interval='month',
            # date_field='created',
            place_id=MIDWEST_STATE_IDS,
        )

pprint(monthly_hist)
pprint(hourly_hist)

exit()



grid = sns.JointGrid(data=df, x='observed_on_mp', y='observed_time_mp', height=10, dropna=True)
grid.ax_marg_x.set_title('Observation times of monarch butterflies in the Midwest US')

# Format X axis labels & ticks
xaxis = grid.ax_joint.get_xaxis()
xaxis.label.set_text('Month')
xaxis.set_major_locator(dates.DayLocator(interval=30))
xaxis.set_major_formatter(dates.DateFormatter('%b %d'))
#xaxis.set_minor_locator(dates.DayLocator(interval=7))
#xaxis.set_minor_formatter(dates.DateFormatter('%d'))

# Format Y axis labels & ticks
yaxis = grid.ax_joint.get_yaxis()
yaxis.label.set_text('Time of Day')
yaxis.set_major_locator(dates.HourLocator(interval=2))
yaxis.set_major_formatter(dates.DateFormatter('%H:%M'))
#yaxis.set_minor_locator(dates.HourLocator())
#yaxis.set_minor_formatter(dates.DateFormatter('%H:%M'))

# Generate a joint plot with marginal plots
# Using the hexbin plotting function, because hexagons are the bestagons.
# Also because it looks just a little like butterfly scales.
grid.plot_joint(plt.hexbin, gridsize=24, cmap=get_colormap(PLOT_COLOR))
grid.plot_marginals(sns.histplot, color=PLOT_COLOR, kde=False)

