import pandas as pd
import os
from config import *
from datetime import datetime
from globals import *


def import_data_orkla_sparebank(directory_path):
    """
    Imports all CSV files from a given directory and concatenates them into a single pandas DataFrame.

    Args:
        directory_path (str): The path to the directory containing the CSV files.

    Returns:
        pd.DataFrame: A single DataFrame containing the concatenated data from all CSV files.
    """
    # List to hold individual DataFrames
    dataframes = []
    encoding="ISO 8859-10"

    # Iterate through all files in the directory
    for file_name in reversed(os.listdir(directory_path)):
        # Check if the file is a CSV file
        if file_name.endswith(".csv"):
            file_path = os.path.join(directory_path, file_name)
            # Load the CSV file into a pandas DataFrame
            df = pd.read_csv(file_path, encoding=encoding, delimiter=';')
            dataframes.append(df)
    
    combined_df = pd.concat(dataframes, ignore_index=True)

    # Replace NaN values with 0 in 'Beløp inn' and 'Beløp ut' columns
    combined_df['Beløp inn'] = combined_df['Beløp inn'].fillna(0)
    combined_df['Beløp ut'] = combined_df['Beløp ut'].fillna(0)

    return combined_df


def analyse(working_df, group_name: str, message_includes_list = [], message_excludes_list = [], includes_dict = {}, excludes_dict = {}):
    
    if not len(message_includes_list) == 0:
        includes = {"Melding/KID/Fakt.nr": message_includes_list}
    else:
        includes = includes_dict

    if not len(message_excludes_list) == 0:
        excludes = {"Melding/KID/Fakt.nr": message_excludes_list}
    else:
        excludes = excludes_dict

    expense_group = ExpenseGroup(includes=includes, excludes=excludes)
    group_df, working_df = expense_group.filter_dataframe(working_df)

    dfprint(group_name, group_df)
    print_metrics(group_name, group_df)

    return group_df, working_df






def dfprint(name,df):
    print(f"\n\n############# {name} ############")
    print(df.shape)  # Check the number of rows and columns
    print(df[['Utført dato', 'Beskrivelse', 'Beløp inn', 'Beløp ut', 'Valuta', 'Melding/KID/Fakt.nr']].head())  # View the first few rows



def print_metrics(name, df):
    print(f"\n\n############# {name} ############")

    total_in = df["Beløp inn"].sum() 
    total_out = df["Beløp ut"].sum()
    print(f"Total in: {total_in}")
    print(f"Total out: {total_out}")
    print(f"Total net flow: {total_in + total_out}")

    days, weeks, months, years = calculate_time(start_time, end_time)
    print(f"Yearly net flow: {(total_in + total_out)/years}")
    print(f"Monthly net flow: {(total_in + total_out)/months}")
    print(f"Weekly net flow: {(total_in + total_out)/weeks}")


def calculate_time(start_date, end_date):

    # Define the two dates
    start_date = datetime(2020, 8, 1)
    end_date = datetime(2025, 1, 9)

    # Calculate the difference in days
    difference = end_date - start_date

    # Convert the difference in days to weeks
    days = difference.days
    weeks = difference.days / 7
    months = difference.days / (365.24/12)
    years = difference.days / 365.24

    return days, weeks, months, years

class ExpenseGroup():

    includes: dict
    excludes: dict
    filtered_dataframe: pd.DataFrame

    def __init__(self, includes ={}, excludes={}):

        self.includes = includes
        self.excludes = excludes


    # def filter_dataframe(self, full_df, exclude=False):

    #     keywords_pattern = "|".join(self.keywords)

    #     # Filter rows where the column contains any of the store names
    #     if not exclude: filtered_df = full_df[full_df["Melding/KID/Fakt.nr"].str.contains(keywords_pattern, case=False, na=False)]
    #     else:           filtered_df = full_df[not full_df["Melding/KID/Fakt.nr"].str.contains(keywords_pattern, case=False, na=False)]

    #     return filtered_df
    


    def filter_dataframe(self, full_df):
        """
        Filters rows in a DataFrame based on column-specific include and exclude keyword lists.

        Parameters:
            full_df (pd.DataFrame): The complete DataFrame to filter.

        Returns:
            pd.DataFrame: A filtered DataFrame based on the include and exclude criteria.
        """

        # Start with the full DataFrame
        filtered_df = full_df.copy()

        # Apply includes
        for column, include_keywords in self.includes.items():
            if column in filtered_df.columns:
                include_pattern = "|".join(include_keywords)  # Create regex for include keywords
                filtered_df = filtered_df[filtered_df[column].str.contains(include_pattern, case=False, na=False)]

        # Apply excludes
        for column, exclude_keywords in self.excludes.items():
            if column in filtered_df.columns:
                exclude_pattern = "|".join(exclude_keywords)  # Create regex for exclude keywords
                filtered_df = filtered_df[~filtered_df[column].str.contains(exclude_pattern, case=False, na=False)]

        remainder_df = full_df.loc[~full_df.index.isin(filtered_df.index)]

        return filtered_df, remainder_df


