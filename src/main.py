import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from datetime import datetime
from globals import *

from utils import *

directory = "data/"
start_time = datetime(2020, 8, 1)
end_time = datetime(2025, 1, 9)

# Import and concatenate all CSV files into a single DataFrame
all_transactions_df = import_data_orkla_sparebank(directory)

# Inspect the combined DataFrame
print(all_transactions_df.shape)  # Check the number of rows and columns
print(all_transactions_df.head())  # View the first few rows


vipps_group = ExpenseGroup(includes={"Undertype":["Straksbetaling"]})
vipps_df,working_df = vipps_group.filter_dataframe(all_transactions_df)


dfprint('VIPPS', vipps_df)
print_metrics('VIPPS', vipps_df)


# List of common Norwegian stores
groceries_group = ExpenseGroup(includes={"Melding/KID/Fakt.nr": ["EXTRA", "REMA", "KIWI", "SPAR", "MENY", "BUNNPRIS"]})
groceries_df,working_df = groceries_group.filter_dataframe(working_df)

dfprint("DAGLIGVARER", groceries_df)

# List of common Norwegian stores
fastfood_group = ExpenseGroup(includes={"Melding/KID/Fakt.nr": [" MCD ", " BK ", " FLY CHICKEN ", " SESAM ", " EGON ", " Wolt ", "Foodora "]})
fast_food_df,working_df = fastfood_group.filter_dataframe(working_df)

dfprint('FAST FOOD', fast_food_df)
print_metrics('FAST FOOD', fast_food_df)


# List of common Norwegian stores
hobby_group = ExpenseGroup(includes={"Melding/KID/Fakt.nr": ["STEAM", "LEGO", "KINO", "Patreon"]})
hobby_df,working_df = fastfood_group.filter_dataframe(working_df)

dfprint('HOBBY', hobby_df)
print_metrics('HOBBY', hobby_df)


# List of common Norwegian stores
fitness_group = ExpenseGroup(includes={"Melding/KID/Fakt.nr": ["STEAM", "LEGO", "KINO", "Patreon"]})
fitness_df,working_df = fastfood_group.filter_dataframe(working_df)

dfprint('HOBBY', hobby_df)
print_metrics('HOBBY', hobby_df)


everything_df,_ = analyse(all_transactions_df, 'EVERYTHING', includes_dict={"Beskrivelse": [""]})


fitness_df,working_df = analyse(working_df, 'FITNESS', includes_dict={"Beskrivelse": ['Sit Studentsams ', "Avtalegiro til Sit", "proteinfabrikken", "PROTEINFABRIK"]})


rent_df,working_df = analyse(working_df, 'RENT', includes_dict={"Beskrivelse": ['Sit Studentsams ', "Avtalegiro til Sit", "proteinfabrikken", "PROTEINFABRIK"]})

savings_df,working_df = analyse(working_df, 'SAVINGS', includes_dict={"Beskrivelse": ["EIKA"]})

