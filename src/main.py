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



everything_df,_ = analyse(all_transactions_df, 'EVERYTHING', includes_dict={"Beskrivelse": [""]})



vipps_df, working_df = analyse(all_transactions_df, 'VIPPS', includes_dict={"Undertype":["Straksbetaling"]})

groceries_df, working_df = analyse(working_df, 'GROCERIES', ["EXTRA", "REMA", "KIWI", "SPAR", "MENY", "BUNNPRIS"])

fastfood_df, working_df = analyse(working_df, 'FAST FOOD', [" MCD ", " BK ", " FLY CHICKEN ", " SESAM ", " EGON ", " Wolt ", "Foodora "])

hobby_df, working_df = analyse(working_df, 'HOBBY', ["STEAM", "LEGO", "KINO", "Patreon"])

fitness_df,working_df = analyse(working_df, 'FITNESS', includes_dict={"Beskrivelse": ['Sit Studentsams ', "Avtalegiro til Sit", "proteinfabrikken", "PROTEINFABRIK"]})

rent_df,working_df = analyse(working_df, 'RENT', ['Leie', 'leie', 'Til: 1210'])

savings_df,working_df = analyse(working_df, 'SAVINGS', includes_dict={"Beskrivelse": ["EIKA"]})

