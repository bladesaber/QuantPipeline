import pandas as pd
import numpy as np
from enum import Enum
from colorama import Fore, Back, Style, init
import arcticdb as adb
from arcticdb import LazyDataFrame


init(autoreset=True)


class ArcticDBReadMode(Enum):
        NORMAL = 1
        LAZY = 2
        QUERY = 3


class ArcticDBSqlUtils(object):
    def __init__(self, path: str):
        self.arctic_db = adb.Arctic(path)
    
    @property
    def list_libraries(self):
        return self.arctic_db.list_libraries()

    def has_library(self, library_name: str) -> bool:
        return self.arctic_db.has_library(library_name)
    
    def has_table(self, library_name: str, table_name: str) -> bool:
        return self.arctic_db.has_symbol(library_name, table_name)

    def list_tables(self, library_name: str) -> list[str]:
        return self.arctic_db.list_symbols(library_name)

    def list_versions(self, library_name: str, table_name: str) -> list[int]:
        library = self.arctic_db.get_library(library_name)
        return library.list_versions(table_name)

    def create_library(self, library_name: str):
        self.arctic_db.create_library(library_name)
        
    @staticmethod
    def generate_query_builder() -> adb.QueryBuilder:
        return adb.QueryBuilder()

    def write_table(
            self, library_name: str, table_name: str, df: pd.DataFrame, version: int = -1, metadata: dict = None
        ):
        assert isinstance(df.index, pd.DatetimeIndex), f'{Fore.RED}Index must be a datetime index{Style.RESET_ALL}'
        assert library_name in self.libraries, f'{Fore.RED}Library {library_name} does not exist{Style.RESET_ALL}'
        library = self.arctic_db.get_library(library_name, create_if_missing=False)
        library.write(
            symbol=table_name, data=df, metadata=metadata,
            prune_previous_versions=version < 0,
            validate_index=True
        )

    def read_table(
            self, library_name: str, table_name: str, version: int = None,
            date_range: tuple[pd.Timestamp, pd.Timestamp] = None,
            columns: list[str] = None,
            mode: ArcticDBReadMode = ArcticDBReadMode.NORMAL,
            query: adb.QueryBuilder = None,
        ) -> pd.DataFrame | LazyDataFrame:
        """if version is None, read the latest version"""
        library = self.arctic_db.get_library(library_name)
        if mode == ArcticDBReadMode.LAZY:
            return library.read(table_name, date_range=date_range, columns=columns, lazy=True, as_of=version)
        elif mode == ArcticDBReadMode.QUERY:
            return library.read(
                table_name, date_range=date_range, columns=columns, query_builder=query, as_of=version
            ).data
        else:
            return library.read(table_name, date_range=date_range, columns=columns, as_of=version).data

    def read_table_metadata(self, library_name: str, table_name: str) -> dict:
        library = self.arctic_db.get_library(library_name)
        return library.read(table_name).metadata

    def read_tables(
            self, 
            library_name: str, table_names: list[str], 
            mode: ArcticDBReadMode = ArcticDBReadMode.NORMAL,
            query: adb.QueryBuilder = None,
        ) -> list[pd.DataFrame | LazyDataFrame]:
        library = self.arctic_db.get_library(library_name)
        if mode == ArcticDBReadMode.LAZY:
            return library.read_batch(table_names, query_builder=None, lazy=True)
        elif mode == ArcticDBReadMode.QUERY:
            return library.read_batch(table_names, query_builder=query, lazy=False).data
        else:
            raise ValueError(f'{Fore.RED}Invalid mode{Style.RESET_ALL}')

    def update_table(
            self, library_name: str, table_name: str, df: pd.DataFrame, append: bool = True,
            date_range: tuple[pd.Timestamp, pd.Timestamp] = None,
        ):
        """
        append/upsert: If True, will write the data even if the symbol does not exist.
        """
        library = self.arctic_db.get_library(library_name)
        library.update(
            symbol=table_name, data=df, upsert=append, prune_previous_versions=True,
            date_range=date_range
        )

    def delete_library(self, library_name: str):
        self.arctic_db.delete_library(library_name)
    
    def delete_table(self, library_name: str, table_name: str):
        library = self.arctic_db.get_library(library_name)
        library.delete(table_name)
    
    def remove_data_of_table(self, library_name: str, table_name: str, date_range: tuple[pd.Timestamp, pd.Timestamp]):
        library = self.arctic_db.get_library(library_name)
        library.delete_data_in_range(table_name, date_range)

    def keep_last_version_table(self, library_name: str, table_name: str):
        self.arctic_db.get_library(library_name).prune_previous_versions(table_name)

    def lazy_df2df(self, lazy_df: LazyDataFrame) -> pd.DataFrame:
        return lazy_df.collect().data


if __name__ == '__main__':
    base_path = 'lmdb:///Users/dng/Desktop/work/arcticdb'
    arctic_db = ArcticDBSqlUtils(base_path)
    print(arctic_db.list_libraries())