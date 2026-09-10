import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import logging
from pathlib import Path

TARGET='time_taken'

# creaate a logger
logger=logging.getLogger('data_preprocessing')
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

def load_data(data_path: Path) -> pd.DataFrame:
    try:
        df=pd.read_csv(data_path)
        return df

    except FileNotFoundError as e:
        logger.error(f'file to load does not exist')
        

def split_data(data: pd.DataFrame, test_size: float, random_state: int):
    try:
        train_data, test_data = train_test_split(
            data, test_size=test_size, random_state=random_state
        )
        logger.info('Data split into train and test successfully')
        return train_data, test_data
    except KeyError:
        logger.error(f'Column {TARGET} not found in data')
        raise
    except Exception as e:
        logger.error(f'Error occurred while splitting the data: {e}')
        raise

def read_params(file_path):
    with open(file_path,'r')as f:
        params_file=yaml.safe_load(f)
        return params_file

def save_data(data:pd.DataFrame, save_path:Path)-> None:
    data.to_csv(save_path, index=False)

if __name__=='__main__':
    # set file paths
    # root path
    root_path=Path(__file__).parent.parent.parent
    # data load path
    data_path=root_path/'data'/'cleaned' / 'swiggy_cleaned.csv'
    # save data directory
    save_data_dir=root_path/'data'/'interim'
    # make it if not exist
    save_data_dir.mkdir(exist_ok=True, parents=True)
    # train and test save paths
    # filenames
    train_filename='train.csv'
    test_filename='test.csv'
    # save path for train and test
    save_train_path=save_data_dir/train_filename
    save_test_path=save_data_dir/test_filename
    # parameter file
    params_file_path=root_path/'params.yaml'

    # load the cleaned data
    data=load_data(data_path)
    logger.info('Data Loaded Successfully')

    # read the parameters
    parameters=read_params(params_file_path)['data_preprocessing']
    test_size=parameters['test_size']
    random_state=parameters['random_state']
    logger.info('parameters read succesfully')

    # split the train and test data
    train_data, test_data=split_data(data, test_size=test_size, random_state=random_state)
    logger. info('Dataset split into train and test data')

    # save the train and test data
    data_subsets=[train_data, test_data]
    data_paths=[save_train_path, save_test_path]
    filename_list=[train_filename, test_filename]
    for filename, path, subset in zip(filename_list, data_paths, data_subsets):
        save_data(data=subset, save_path=path)
        logger.info(f"{filename.replace('.csv','')} data saved to location")
        