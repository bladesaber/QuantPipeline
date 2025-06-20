import pickle
import h5py
from typing import Dict, Literal
import pandas as pd
import numpy as np


class SaveLoad(object):
    @staticmethod
    def save_pickle(obj, path: str):
        with open(path, 'wb') as f:
            pickle.dump(obj, f)

    @staticmethod
    def load_pickle(path: str):
        with open(path, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def save_h5(
        obj_dict: Dict[str, pd.DataFrame], path: str, 
        compression: Literal['gzip', 'lzf', 'szip', 'none'] = 'gzip', compression_level: int = 4
        ):
        '''
        Save DataFrames to HDF5 file with compression
        
        Args:
            obj_dict: Dictionary of DataFrames to save
            path: Path to save the HDF5 file
            compression: Compression method to use, default is gzip
            compression_level: Compression level (0-9, higher means better compression but slower)
        '''
        with h5py.File(path, 'w') as f:
            for name, df in obj_dict.items():
                group_write = f.create_group(name)
                for col_name, col_dtype in zip(df.columns, df.dtypes):
                    if col_dtype == 'object':
                        dset = group_write.create_dataset(
                            col_name, 
                            (len(df[col_name]),), 
                            dtype=h5py.string_dtype(encoding='utf-8'),
                            compression=compression,
                            compression_opts=compression_level
                        )
                        dset[:] = df[col_name]
                        dset.attrs['dtype'] = 'object'
                    elif col_dtype == 'datetime64[ns]':
                        # Convert datetime64 to timestamps (nanoseconds since epoch)
                        timestamps = df[col_name].astype(np.int64)
                        dset = group_write.create_dataset(
                            col_name, 
                            data=timestamps,
                            compression=compression,
                            compression_opts=compression_level
                        )
                        dset.attrs['dtype'] = 'datetime64[ns]'
                    else:
                        dset = group_write.create_dataset(
                            col_name, 
                            data=df[col_name],
                            compression=compression,
                            compression_opts=compression_level
                        )
                        dset.attrs['dtype'] = str(col_dtype)

    @staticmethod
    def load_h5(path: str):
        with h5py.File(path, 'r') as f:
            result = {}
            for name, group in f.items():
                df_dict = {}
                for col_name, dataset in group.items():
                    dtype = dataset.attrs.get('dtype', '')
                    if dtype == 'object':
                        # Convert byte array to string array
                        df_dict[col_name] = [item.decode('utf-8') for item in dataset[:]]
                    elif dtype == 'datetime64[ns]':
                        # Convert timestamps back to datetime64
                        df_dict[col_name] = pd.to_datetime(dataset[:])
                    else:
                        df_dict[col_name] = dataset[:]
                result[name] = pd.DataFrame(df_dict)
            return result

    @staticmethod
    def print_h5(path: str, pre=''):
        with h5py.File(path, 'r') as f:
            for key, val in f.items():
                if isinstance(val, h5py.Group):
                    print(pre + '└── ' + key)
                    SaveLoad.print_h5(val, pre + '    ')
                else:
                    print(pre + '├── ' + key)

