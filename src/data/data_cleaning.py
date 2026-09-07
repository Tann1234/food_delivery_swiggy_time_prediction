# Import some library
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# create logger
logger=logging.getLogger('data_cleaning')
logger.setLevel(logging.INFO)

# console handler
handler=logging.StreamHandler()
handler.setLevel(logging.INFO)

# add handler to logger
logger.addHandler(handler)

# create a formatter
formatter=logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# add formatter to handler
handler.setFormatter(formatter)


columns_to_drop=['rider_id',
                 'restaurant_latitude',
                 'restaurant_longitude',
                 'delivery_latitude',
                 'delivery_longitude',
                 'order_date',
                 'order_time_hour',
                 'order_day', 
                'city_name',
                'order_day_of_week',
                'order_month']

def load_data(data_path:Path) -> pd.DataFrame:
    try:
        df=pd.read_csv(data_path)
        logger.info('swiggy data loaded succcesfully')
        return df
    except FileNotFoundError as e:
        logger.error(f'file not found :{e}')
    except pd.errors.EmptyDataError as e:
        logger.error(f'data type error:{e}')
    except Exception as e:
        logger.error(f'Error loading data :{e}')
        raise

def change_column_name(data:pd.DataFrame):
    try:
        logger.info('Starting column renaming process')
        renamed_data=(
             data.rename(str.lower, axis=1)
            .rename({
                'delivery_person_id': 'rider_id',
                'delivery_person_age': 'age',
                'delivery_person_ratings': 'ratings',
                'delivery_location_latitude': 'delivery_latitude',
                'delivery_location_longitude': 'delivery_longitude',
                'time_orderd': 'order_time',
                'time_order_picked': 'order_picked_time',
                'weatherconditions': 'weather',
                'road_traffic_density': 'traffic',
                'city': 'city_type',
                'time_taken(min)': 'time_taken'},
                axis=1)
        )
        logger.info('Column reanaming completed')
        logger.debug(f"Columns after renaming: {renamed_data.columns.tolist()}")
        return renamed_data
    except KeyError as e:
        logger.error(f'column not found {e}')
    except IndentationError as e:
        logger.error(f'indentation error {e}')
    except Exception as e:
        logger.error(f"Error during column renaming: {e}")
        raise


def data_cleaning(data:  pd.DataFrame):
    try:
        logger.info('Starting data cleaning process')

        # filter out minors and invalid rating 
        minor_data=data.loc[data['age'].astype('float')<18]
        minor_index=minor_data.index.tolist()
        six_star_data=data.loc[data['ratings']=='6']
        six_star_index=six_star_data.index.tolist()
        cleaned_data=(
            data
            .drop(columns='id')
            .drop(index=minor_index)
            .drop(index=six_star_index)
            .replace('NaN ', np.nan)
            .assign(
                # city column out of rider_id
                city_name=lambda x: x['rider_id'].str.split('RES').str.get(0),
                age=lambda x: x['age'].astype('float'),
                ratings=lambda x: x['ratings'].astype('float'),
                restaurant_latitude=lambda x: x['restaurant_latitude'].abs(),
                restaurant_longitude=lambda x: x['restaurant_longitude'].abs(),
                delivery_latitude=lambda x: x['delivery_latitude'].abs(),
                delivery_longitude=lambda x: x['delivery_longitude'].abs(),
                order_date=lambda x: pd.to_datetime(x['order_date'], dayfirst=True),
                order_day=lambda x: x['order_date'].dt.day,
                order_month=lambda x: x['order_date'].dt.month,
                order_day_of_week=lambda x: x['order_date'].dt.day_name().str.lower(),
                is_weekend=lambda x: x['order_date'].dt.day_name().isin(['Saturday', 'Sunday']).astype(int),
                # time based columns
                order_time=lambda x: pd.to_datetime(x['order_time'], format='mixed'),
                order_picked_time=lambda x: pd.to_datetime(x['order_picked_time'], format='mixed'),
                # time taken to pick order
                pickup_time_minutes=lambda x:((x['order_picked_time']-x['order_time']).dt.seconds/60),

                # hour in which order was placed
                order_time_hour=lambda x: x['order_time'].dt.hour,

                #time of the day when order was placed
                order_time_of_day=lambda x:(x['order_time_hour'].pipe(time_of_day)),

                # categorical columns
                weather=lambda x:(x['weather'].str.split(' ').str.get(1).str.lower().replace('NaN', np.nan)),
                traffic=lambda x: x['traffic'].str.rstrip().str.lower(),
                type_of_order=lambda x: x['type_of_order'].str.rstrip().str.lower(),
                type_of_vehicle=lambda x: x['type_of_vehicle'].str.rstrip().str.lower(),
                festival= lambda x: x['festival'].str.rstrip().str.lower(),
                city_type= lambda x: x['city_type'].str.rstrip().str.lower(),
                multiple_deliveries=lambda x: x['multiple_deliveries'].astype(float),
                time_taken=lambda x: (x['time_taken'].str.split(' ').str.get(1).astype('int')))
                .drop(columns=['order_time', 'order_picked_time'])
        
            )
        logger.info('Data cleaning completed succesfully')
        logger.debug(f"Cleaned DataFrame shape: {cleaned_data.shape}")
        return cleaned_data
    except IndexError as e:
        logger.error(f"Error occurred while cleaning data: {e}")
    except Exception as e:
        logger.error(f'Error during data cleaning {e}')
        raise


def clean_lat_long(data:pd.DataFrame, threshold=1):
    try:
        logger.info('Starting latitude and longitude cleaning process')
        location_columns = [
            'restaurant_latitude',
            'restaurant_longitude',
            'delivery_latitude',
            'delivery_longitude'
        ]

        cleaned_data=(
        data.assign(**{
            col:(
                np.where(data[col]< threshold, np.nan, data[col].values)
            )
            for col in location_columns
        })
    )
        logger.info('Latitude and longitude cleaning completed successfully')
        logger.debug(f"Applied threshold {threshold} to columns: {location_columns}")
        return cleaned_data
    except Exception as e:
        logger.error(f"Error during latitude/longitude cleaning: {e}")
        raise

# extract day, day name, month and year
def extract_datetime_features(ser):
    try:
        logger.info('Starting extraction of datetime features')
        date_col=pd.to_datetime(ser, dayfirst=True)
        logger.info('Datetime features extraction completed successfully')
        return(
            pd.DataFrame(
                {'day': date_col.dt.day,
                'month': date_col.dt.month,
                'year': date_col.dt.year,
                'day_of_week': date_col.dt.day_name(),
                'is_weekend': date_col.dt.day_name().isin(['Saturday', 'Sunday']).astype(int)
                }
            ))
    except Exception as e:
        logger.error(f"Error during datetime feature extraction: {e}")
        raise
             
def time_of_day(ser):
    try:
        logger.info('Starting time of day categorization')
        time_col=pd.to_datetime(ser,format='mixed').dt.hour
        logger.info('Time of day categorization completed successfully')
        return(
            pd.cut(time_col,bins=[0,6,12,17,20,24],right=True,
                   labels=["after_midnight","morning","afternoon","evening","night"])
        )
    except Exception as e:
        logger.error(f"Error during time of day categorization: {e}")
        raise

def calculate_haversine_distance(data:pd.DataFrame):
    try:
        logger.info('Starting Haversine distance calculation')
        location_columns = ['restaurant_latitude',
                            'restaurant_longitude',
                            'delivery_latitude',
                            'delivery_longitude']

        lat1 = data['restaurant_latitude']
        lon1 = data['restaurant_longitude']
        lat2 = data['delivery_latitude']
        lon2 = data['delivery_longitude']

        lon1, lat1, lon2, lat2=map(np.radians, [lon1, lat1, lon2, lat2])

        dlon=lon2-lon1
        dlat=lat2-lat1

        a=np.sin(
            dlat/2.0)**2 +np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
        c=2*np.arcsin(np.sqrt(a))
        distance=6371*c
        data_distance=(data.assign(
            distance=distance))
        logger.info('Haversine distance calculation completed successfully')
        logger.debug(f"Calculated distances for {len(data_distance)} records")
        return data_distance
    except ValueError as e:
        logger.error(f"Value error during Haversine distance calculation: {e}")
    except TypeError as e:
        logger.error(f"Type error during Haversine distance calculation: {e}")
    except Exception as e:
        logger.error(f"Error during Haversine distance calculation: {e}")   
        raise

def create_distance_type(data:pd.DataFrame):
    try:
        logger.info('Starting distance type categorization')

        data_type_distance=(
        data.assign(
            distance_type=pd.cut(data['distance'], bins=[0,5,10,15,25],
            right=False, labels=['short', 'medium', 'long', 'verylong']
            ))
            )
        logger.info('Distance type categorization completed successfully')
        logger.debug(f"Distance type distribution:\n{data_type_distance['distance_type'].value_counts()}")
        return data_type_distance
    except ValueError as e:
        logger.error(f"Value error during distance type categorization: {e}")       
        raise

def drop_columns(data:pd.DataFrame, columns:list) -> pd.DataFrame:
    try:
        logger.info(f'Starting to drop column {columns}')
        df=data.drop(columns=columns)
        logger.info(f'columns {columns} dropped successfully')
        return df
    except Exception as e:
        logger.error(f"Error during column dropping: {e}")
        raise

def perform_data_cleaning(data: pd.DataFrame, saved_data_path='swiggy_cleaned.csv'):
    try:
        logger.info("Starting full data cleaning pipeline...")

        cleaned_data = (
            data.pipe(change_column_name)
            .pipe(data_cleaning)
            .pipe(clean_lat_long)
            .pipe(calculate_haversine_distance)
            .pipe(create_distance_type)
            .pipe(drop_columns,columns=columns_to_drop)  # drop after they exist
            )


        # Save the cleaned dataset
        cleaned_data.to_csv(saved_data_path, index=False)
        logger.info(f"Data cleaning pipeline completed successfully. File saved at: {saved_data_path}")
        logger.debug(f"Final cleaned DataFrame shape: {cleaned_data.shape}")

    except Exception as e:
        logger.error(f"Error during full data cleaning pipeline: {e}")
        raise

    # save the data

if __name__=='__main__':
    # data path for data
    root_path=Path(__file__).parent.parent.parent
    # data save directory
    cleaned_data_save_dir=root_path/'data'/'cleaned'
    # make directory if not exists
    cleaned_data_save_dir.mkdir(exist_ok=True, parents=True)
    # cleaned data file name
    cleaned_data_filename='swiggy_cleaned.csv'
    # data save path
    cleaned_data_save_path=cleaned_data_save_dir/cleaned_data_filename
    # data load path
    data_load_path=root_path/'data'/'raw'/'swiggy.csv'

    # load data
    df=load_data(data_load_path)
    perform_data_cleaning(data=df, saved_data_path=cleaned_data_save_path)
