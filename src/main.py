import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from datetime import datetime
from globals import *

from utils import *

directory = "data/"


# Import and concatenate all CSV files into a single DataFrame
all_transactions_df = import_data_orkla_sparebank(directory)

# # Inspect the combined DataFrame
# print(all_transactions_df.shape)  # Check the number of rows and columns
# print(all_transactions_df.head())  # View the first few rows


accountant = Accountant(all_transactions_df)

accountant.add_expense_group('BANK INTEREST', includes_dict={'Undertype':['Renter']})

accountant.add_expense_group('SALARY', includes_dict={'Undertype':['Lønn', 'Sumpost OCR m/underspesifisering']})

accountant.add_expense_group('VIPPS', includes_dict={"Undertype":["Straksbetaling", "Kreditoverføring"]})

accountant.add_expense_group('GROCERIES', ["EXTRA", "REMA", "KIWI", "SPAR", "MENY", "BUNNPRIS", "COOP", "EUROSPAR", "OBS", "MATKROKEN", "JOKER", "BILKA", "RAPIDO", "SITO"])

accountant.add_expense_group('FAST FOOD', [" MCD ", " BK ", " FLY CHICKEN ", " SESAM ", " EGON ", " Wolt ", "Foodora ", "DOMINOS", "PIZZABAKER", "SUBWAY", "KFC", "PEPPES", "SUSHI", "NARVESEN", "CIRCLE K FOOD", "SABRURA", "GORDITA"])

accountant.add_expense_group('HOBBY', ["STEAM","EPIC GAMES", "PLAYSTATION", "LEGO", "KINO", "Patreon", "FOTO", "PHOTO", "OUTLAND"])

accountant.add_expense_group('FITNESS', includes_dict={"Beskrivelse": ['Sit Studentsams ', "Avtalegiro til Sit", "proteinfabrikken", "PROTEINFABRIK", "GYM", "SATSELIXIA", "FRESH FITNESS", "XXL", "INTERSPORT", "SPORT1", "SPORT OUTLET", "GYMSHARK"]})

accountant.add_expense_group('RENT', message_includes_list=["Husleie", 'Leie', 'leie', 'Til: 1210', "Strindvegen", "STRINDVEGEN", "Avtalegiro til STUDENTSAMSKIPNADEN"], message_excludes_list=['Depositum'])

accountant.add_expense_group('SAVINGS', includes_dict={"Beskrivelse": ["EIKA", "Boligsparing for ungdom"]})

accountant.add_expense_group('CLOTHES', ['Zalando', "CUBUS", "ZARA", "CARLINGS", "WEEKDAY", "DRESSMANN", "SKORINGEN", "ECCO", "MATCH", "JACK&JONES", "HILFIGER", "VOLT", "BERTONI"])

accountant.add_expense_group('AKADEMIKA', ['AKADEMIKA'])

accountant.add_expense_group('POLET', ['POLET'])

accountant.add_expense_group('TRANSPORT', ['SAS', 'NORWEGIAN', "ATB", "RUTER", "FLYBUSS", "VY", "SJ NORD", "TAXI", "FLYTOG"])

accountant.add_expense_group('OPPKJØRING', ["TABS", "STATENS VEGVESEN"])

accountant.add_expense_group('TECH', ["EKJOP", "POWER", "APPLE", "PROSHOP", "KOMPLETT"])

accountant.add_expense_group('INTERIOR', ["KITCHN", "TILBORDS", "IKEA", "CLAS", "PRINCESS", "RUSTA", "EUROPRIS"])

accountant.add_expense_group('HEALTH AND CARE', ["APOTEK", "VITUSAPOTEK", "BOOTS", "LEGE", "OPTIKER", "SYKEHUS"])

# accountant.add_expense_group('STUDENT LOAN', ["STATENS LÅNEKASSE"])
accountant.add_expense_group('STUDENT LOAN', includes_dict={'Beskrivelse':["STATENS L"]})
accountant.add_expense_group('TAX CORRECTION', includes_dict={'Beskrivelse':["Skatteetaten"]})

# accountant.add_expense_group('SALARY', ["HYDRO", "LØNN", "RENNEBU KIRKELIGE FELLESRÅD", "NTNU", "Fra: NORGES TEKNISK", "NORGES TEKN.NATURVITENSK.UNIVE", "Fra: Soknedal Bakeri", "7288 Soknedal"])



accountant.process_transactions()
accountant.print_expenses_report()


accountant.plot_expenses_pie_chart()
accountant.plot_expenses_bar_chart()



# accountant.analyze_transaction_network(min_word_length=3, threshold= 5,remove_percent= 0)   
# accountant.plot_remaining_expenses_pie_chart(top_percent=100)