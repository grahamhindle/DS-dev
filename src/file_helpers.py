import pandas as pd
import os


def read_to_df(file_path):
    """Read on-disk data and return a dataframe.

    Positional argument:
        file_path -- a well formed csv datafile

    Requirements:
        This function depends upon the pandas function read_csv
        file exists - an error is thrown if file_path does not exist"""

    try:
        os.path.exists(file_path)
        df = pd.read_csv(file_path)
    except IOError:
        print('File :' + file_path + ' cannot be found - please check path and filename')
    else:
        assert isinstance(df, object)
        return df

def select_columns(data_frame, column_names):
    """Return a subset of a data frame by column names.

        Positional arguments:
            data_frame -- a pandas DataFrame object
            column_names -- a list of column names to select

        Example:
            data = read_to_df('train.csv')
            selected_columns = ['SalePrice', 'GrLivArea', 'YearBuilt']
            sub_df = select_columns(data, selected_columns)
        Error handling:
            returns empty data frame if invalid filename passed or one
            or more columns does not exist
        """
    try:
        data_frame is not None
        if all([col in data_frame.columns for col in column_names]):  # check for valid columns
            return data_frame[column_names].copy(deep=True)  # deep copy to enable modification
    except:
        pass

def column_cutoff(data_frame, cutoffs):
    """Subset data frame by cutting off limits on column values.

    Positional arguments:
        data -- pandas DataFrame object
        cutoffs -- list of tuples in the format:
        (column_name, min_value, max_value)

    Example:
        data_frame = read_into_data_frame('train.csv')
        # Remove data points with SalePrice < $50,000
        # Remove data points with GrLiveAre > 4,000 square feet
        cutoffs = [('SalePrice', 50000, 1e10), ('GrLivArea', 0, 4000)]
        selected_data = column_cutoff(data_frame, cutoffs)
    """
    index_names = []

    dftemp = data_frame.copy()  # copy dataframe to preserve original
    if dftemp is not None:  # check dataframe is not empty
        for x in range(len(cutoffs)):
            column_name = cutoffs[x][0]  # extract column name
            min_value = float(cutoffs[x][1])  # extract minimum value
            max_value = float(cutoffs[x][2])  # extract maximum value
            index_names.extend(dftemp[(dftemp[column_name] < min_value) |
                                      (dftemp[
                                           column_name] > max_value)].index)  # create index of all values that meet condition
        dftemp.drop(index_names, inplace=True)

    return dftemp
