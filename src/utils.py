import pandas as pd
import os
from config import *
from datetime import datetime
from globals import *
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import combinations
from sklearn.feature_extraction.text import CountVectorizer
import re
import networkx as nx
from collections import Counter
import numpy as np
from matplotlib.cm import get_cmap





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
    global message_column

    # Iterate through all files in the directory
    for file_name in reversed(os.listdir(directory_path)):
        # Check if the file is a CSV file
        if file_name.endswith(".csv"):
            file_path = os.path.join(directory_path, file_name)
            # Load the CSV file into a pandas DataFrame
            df = pd.read_csv(file_path, encoding=encoding, delimiter=';')
            dataframes.append(df)
    
    combined_df = pd.concat(dataframes, ignore_index=True)

    
    KID_indicators = ["KID", "Melidng", "MELDING", "Message", "MESSAGE"]
    KID_pattern = "|".join(KID_indicators)  # Create regex for include keywords

    for column_name in df.columns:
        if any(keyword in column_name for keyword in KID_indicators):
            message_column = column_name
            print(f"Message column identified as '{message_column}'")
            break
        


    # Replace NaN values with 0 in 'Beløp inn' and 'Beløp ut' columns
    combined_df['Beløp inn'] = combined_df['Beløp inn'].fillna(0)
    combined_df['Beløp ut'] = combined_df['Beløp ut'].fillna(0)

    combined_df.dropna(subset=[message_column], inplace=True)

    return combined_df


def analyse(working_df, group_name: str, message_includes_list = [], message_excludes_list = [], includes_dict = {}, excludes_dict = {}):
    
    if not len(message_includes_list) == 0:
        includes = {message_column: message_includes_list}
    else:
        includes = includes_dict

    if not len(message_excludes_list) == 0:
        excludes = {message_column: message_excludes_list}
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
    #     if not exclude: filtered_df = full_df[full_df[message_column].str.contains(keywords_pattern, case=False, na=False)]
    #     else:           filtered_df = full_df[not full_df[message_column].str.contains(keywords_pattern, case=False, na=False)]

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



class Accountant():

    transactions_df:    pd.DataFrame
    expense_groups:     dict
    expense_dfs:        dict
    remainder_df:       pd.DataFrame

    def __init__(self, transactions: pd.DataFrame):

        self.transactions_df = transactions
        self.expense_groups = {}
        self.expense_dfs    = {}

    def add_expense_group(self, group_name: str, message_includes_list = [], message_excludes_list = [], includes_dict = {}, excludes_dict = {}):

        if not len(message_includes_list) == 0:
            includes = {message_column: message_includes_list}
        else:
            includes = includes_dict

        if not len(message_excludes_list) == 0:
            excludes = {message_column: message_excludes_list}
        else:
            excludes = excludes_dict

        self.expense_groups[group_name] = ExpenseGroup(includes=includes, excludes=excludes)


    def process_transactions(self):

        working_df = self.transactions_df.copy(deep=True)

        for group in self.expense_groups:

            expense_group = self.expense_groups[group]
            self.expense_dfs[group], working_df = expense_group.filter_dataframe(working_df)

        self.remainder_df = working_df


    def print_expenses_report(self):

        for expense_type in self.expense_dfs:

            print_metrics(expense_type, self.expense_dfs[expense_type])


    def plot_expenses_pie_chart(self):
        """
        Plots two pie charts: one for money in (earnings) and one for money out (expenses),
        showing percentage-only labels and including an 'Other' segment for remaining items.
        """
        # Dictionaries to store totals for money in and money out
        money_in_totals = {}
        money_out_totals = {}

        # Categorize each expense group
        for group_name, df in self.expense_dfs.items():
            
            net_income = 0
            if 'Beløp inn' in df.columns:
                net_income += df['Beløp inn'].sum()
            
            if 'Beløp ut' in df.columns:
                net_income += df['Beløp ut'].sum()
                
            if net_income > 0:
                money_in_totals[group_name] = net_income
            elif net_income < 0:
                money_out_totals[group_name] = -net_income

        # Calculate totals for remaining transactions
        if self.remainder_df is not None:
            if 'Beløp inn' in self.remainder_df.columns:
                other_in_total = self.remainder_df['Beløp inn'].sum()
                if other_in_total > 0:
                    money_in_totals['Other'] = other_in_total

            if 'Beløp ut' in self.remainder_df.columns:
                other_out_total = -self.remainder_df['Beløp ut'].sum()
                if other_out_total > 0:
                    money_out_totals['Other'] = other_out_total

        # Prepare data for pie charts
        in_labels = list(money_in_totals.keys())
        in_values = list(money_in_totals.values())
        out_labels = list(money_out_totals.keys())
        out_values = list(money_out_totals.values())

        # Create two pie charts in the same figure
        fig, axs = plt.subplots(1, 2, figsize=(14, 7))

        # Create a colormap instance (e.g., 'tab20c')
        cmap = get_cmap('tab20c')
        colors = [cmap(i / len(out_values)) for i in range(len(out_values))]

        # Money In Pie Chart
        axs[0].pie(
            in_values,
            labels=in_labels,
            autopct='%1.1f%%',
            colors=colors,
            startangle=140
        )
        axs[0].set_title('Money In (Earnings)')
        axs[0].axis('equal')  # Equal aspect ratio ensures the pie is a circle


        # Money Out Pie Chart
        axs[1].pie(
            out_values,
            labels=out_labels,
            autopct='%1.1f%%',
            colors=colors,
            startangle=140
        )

        axs[1].set_title('Money Out (Expenses)')
        axs[1].axis('equal')  # Equal aspect ratio ensures the pie is a circle

        # Show the plot
        plt.suptitle('Expense Distribution: Money In vs Money Out', fontsize=16)
        plt.tight_layout()
        plt.show()



    def plot_expenses_bar_chart(self, date_column='Utført dato'):
        """
        Plots a stacked bar chart showing monthly expenses for each expense group.
        Automatically excludes groups that do not have any expenses.
        
        Parameters:
        - date_column (str): The name of the column containing date information.
        """
        plt.close('all')  # Close any open figures to avoid duplicate plots

        # Ensure the expense_dfs dictionary is present
        if not hasattr(self, 'expense_dfs') or not self.expense_dfs:
            print("No expense data available.")
            return

        # Create an empty DataFrame to store combined data from all expense groups
        all_expenses = pd.DataFrame()

        # Iterate over each group in expense_dfs
        for group_name, df in self.expense_dfs.items():
            if date_column not in df.columns or 'Beløp ut' not in df.columns:
                continue

            # Convert the date column to datetime format
            df[date_column] = pd.to_datetime(df[date_column], format='%d.%m.%Y', errors='coerce')

            # Filter for expenses only (negative 'Beløp ut') and take absolute values
            expense_df = df[df['Beløp ut'] + df['Beløp inn'] < 0].copy()
            expense_df['Abs Beløp ut'] = expense_df['Beløp ut'].abs()

            # Skip groups with no expenses
            if expense_df.empty:
                continue

            # Add a new column for the month-year (e.g., "Jan 2022")
            expense_df['Month-Year'] = expense_df[date_column].dt.to_period('M')

            # Group by Month-Year and sum expenses
            grouped_expenses = expense_df.groupby('Month-Year')['Abs Beløp ut'].sum().reset_index()
            grouped_expenses.rename(columns={'Abs Beløp ut': group_name}, inplace=True)

            # Merge with the main DataFrame
            if all_expenses.empty:
                all_expenses = grouped_expenses
            else:
                all_expenses = pd.merge(all_expenses, grouped_expenses, on='Month-Year', how='outer')

        # Check if there are any expenses to plot
        if all_expenses.empty:
            print("No expenses to plot.")
            return

        # Fill NaN values with 0 (for months without data in specific groups)
        all_expenses.fillna(0, inplace=True)

        # Sort by Month-Year
        all_expenses.sort_values(by='Month-Year', inplace=True)

        # Convert Month-Year to string for better readability in the plot
        all_expenses['Month-Year'] = all_expenses['Month-Year'].astype(str)

        # Plot the stacked bar chart
        ax = all_expenses.set_index('Month-Year').plot(
            kind='bar',
            stacked=True,
            figsize=(12, 6),
            colormap='tab20c',  # Use a large colormap to minimize duplicate colors
            width=0.8
        )

        # Customize the plot
        ax.set_title('Monthly Expenses by Expense Group', fontsize=16)
        ax.set_xlabel('Month-Year', fontsize=12)
        ax.set_ylabel('Total Expenses (NOK)', fontsize=12)
        ax.legend(title='Expense Groups', bbox_to_anchor=(1.05, 1), loc='upper left')

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')

        plt.tight_layout()  # Adjust layout for better spacing
        plt.show()




    def should_merge_nodes(self, node, neighbor, G, messages):
        """
        Determines whether two nodes should be merged based on:
        - Shared neighbors with identical edge weights.
        - If the two nodes are isolated pairs (i.e., connected only to each other).
        
        Parameters:
            node (str): The first node.
            neighbor (str): The second node.
            G (networkx.Graph): The graph where nodes and edges exist.
            messages (list): A list of transaction messages to check word order for preserving correct order.
        
        Returns:
            bool: True if nodes should be merged, False otherwise.
        """
        # Check if both nodes have identical neighbors (i.e., they always co-occur)
        neighbors = list(G.neighbors(node))
        neighbor_neighbors = list(G.neighbors(neighbor))
        
        # Check for identical neighbors
        common_neighbors = set(neighbors) & set(neighbor_neighbors)
        
        if common_neighbors == set(neighbors):  # They have identical neighbors
            # Check that all edge weights between the nodes and their common neighbors are the same
            weights_match = True
            for common_neighbor in common_neighbors:
                weight1 = G[node][common_neighbor]['weight']
                weight2 = G[neighbor][common_neighbor]['weight']
                if weight1 != weight2:
                    weights_match = False
                    break
            
            if weights_match:
                return True, None  # Return True to merge, without specific name yet
        
        # Check for isolated nodes (nodes that are only connected to each other)
        if len(neighbors) == 1 and len(neighbor_neighbors) == 1 and neighbors[0] == neighbor:
            # If both nodes are connected only to each other, they should be merged
            return True, None  # Return True to merge, without specific name yet

        return False, None


    def merge_nodes(self, node, neighbor, G, merged_nodes, messages):
        """
        Perform the actual merging of two nodes into a single new node in the graph.
        Decide on the new node's name, based on order and eliminating duplicates.
        
        Parameters:
            node (str): The first node to merge.
            neighbor (str): The second node to merge.
            G (networkx.Graph): The graph where nodes are merged.
            merged_nodes (dict): A dictionary tracking the merged nodes.
            messages (list): A list of transaction messages to check word order for preserving correct order.
        
        Returns:
            None: The function modifies the graph in place.
        """
        # Decide the new node name based on the alphabetical order or message order
        if node != neighbor:
            new_node = f"{min(node, neighbor)} {max(node, neighbor)}"
        else:
            new_node = node  # If the nodes are identical, no merging needed

        example_message = next(
            (message for message in messages 
            if node.lower() in message.lower().split() and neighbor.lower() in message.lower().split()), 
            None
        )

        if example_message is not None:
            word_order = example_message.split()
            node_position = next(i for i, word in enumerate(word_order) if word.lower() == node.lower())
            neighbor_position = next(i for i, word in enumerate(word_order) if word.lower() == neighbor.lower())
            
            if node_position > neighbor_position:
                # If the node appears after the neighbor in the message, swap them
                new_node = f"{neighbor} {node}"
            else:
                new_node = f"{node} {neighbor}"
        
        # Ensure no duplicate words in the new node name
        words = new_node.split()
        new_node = " ".join(sorted(set(words), key=words.index))  # This keeps the original order and removes duplicates

        # Merge the nodes in the graph
        if node not in merged_nodes:
            merged_nodes[node] = new_node
        if neighbor not in merged_nodes:
            merged_nodes[neighbor] = new_node
        
        # Replace the old nodes with the new merged node
        for neighbor_of_node in list(G.neighbors(node)):
            G.add_edge(new_node, neighbor_of_node, weight=G[node][neighbor_of_node]['weight'])
        
        for neighbor_of_neighbor in list(G.neighbors(neighbor)):
            G.add_edge(new_node, neighbor_of_neighbor, weight=G[neighbor][neighbor_of_neighbor]['weight'])
        
        # Remove the original nodes from the graph
        G.remove_node(node)
        G.remove_node(neighbor)


    def analyze_transaction_network(self, min_word_length=2, threshold=5, remove_percent=10):
        """
        Analyzes the co-occurrence network of words in transaction messages, merges nodes with always co-occurring edges,
        and visualizes the final graph. The merging process continues iteratively until no further merges are possible.

        Parameters:
            threshold (int): Minimum co-occurrence count to display an edge in the network.
            remove_percent (int): Percentage of the most common words to remove from the network.
        
        Returns:
            None
        """
        if self.remainder_df is None or message_column not in self.remainder_df.columns:
            print(f"No remaining transactions to analyze or \'{message_column}\' column missing.")
            return

        # Tokenize and clean messages
        messages = self.remainder_df[message_column].fillna("").astype(str)
        amounts = self.remainder_df['Beløp ut'].abs()

        # Initialize a co-occurrence dictionary
        co_occurrence = defaultdict(lambda: defaultdict(int))
        all_tokens = []  # List to keep track of all tokens for frequency calculation
        exclude_words = []

        # Tokenize each message
        for idx, message in enumerate(messages):
            # Clean and split message into words
            tokens = re.findall(r'\b\w+\b', message.lower())  # Extract words only
            tokens = [token for token in tokens if len(token) > min_word_length and not token in exclude_words]  # Remove short tokens
            all_tokens.extend(tokens)

            # Increment co-occurrence count for each pair of tokens
            for i in range(len(tokens)):
                for j in range(i + 1, len(tokens)):  # Only consider pairs (i, j) where i < j
                    co_occurrence[tokens[i]][tokens[j]] += 1
                    co_occurrence[tokens[j]][tokens[i]] += 1  # Undirected graph

        # Calculate word frequencies (for removal of top x% most common words)
        word_counts = Counter(all_tokens)
        total_words = sum(word_counts.values())

        # Sort words by frequency and determine the threshold for removal
        sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        remove_count = int((remove_percent / 100) * total_words)

        # Get the top x% most common words to remove
        most_common_words = set(word for word, _ in sorted_word_counts[:remove_count])

        # Build the graph using networkx
        G = nx.Graph()

        # Add nodes and edges based on co-occurrence, no filtering yet
        for word1 in co_occurrence:
            for word2 in co_occurrence[word1]:
                weight = co_occurrence[word1][word2]
                if weight >= threshold:  # Only include edges with co-occurrence above the threshold
                    G.add_edge(word1, word2, weight=weight)

        # Remove the top x% most common words from the graph (remove nodes)
        nodes_to_remove = [node for node in G.nodes if node in most_common_words]
        G.remove_nodes_from(nodes_to_remove)

        # Remove nodes that are numbers
        nodes_to_remove = [node for node in G.nodes if node.isdigit()]
        G.remove_nodes_from(nodes_to_remove)

        # Iteratively merge nodes until no further merges are possible
        changes = True
        merged_nodes = {}  # To track merged nodes
        while changes:
            changes = False  # To track if any merge occurred in this iteration

            # Try to merge nodes based on shared neighbors and edge weights
            for node in list(G.nodes):  # Using list to allow modification during iteration
                if node not in merged_nodes:  # If the node has not been merged already
                    # Find neighbors of the current node
                    neighbors = list(G.neighbors(node))

                    for neighbor in neighbors:
                        if node != neighbor:  # Skip self-loop
                            # Check if nodes should be merged
                            should_merge, _ = self.should_merge_nodes(node, neighbor, G, messages)
                            
                            if should_merge:
                                # Merge the nodes
                                self.merge_nodes(node, neighbor, G, merged_nodes, messages)
                                changes = True

        self.G = G

        '''
        # Visualize the updated network after merging nodes
        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(G, k=0.15, iterations=20)  # Positioning of nodes
        nx.draw_networkx_nodes(G, pos, node_size=50, node_color="skyblue", alpha=0.7)
        nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
        nx.draw_networkx_labels(G, pos, font_size=12, font_family="sans-serif")
        
        plt.title("Co-occurrence Network with Merged Nodes (Iterative Merging)")
        plt.axis('off')
        plt.show()
        '''



    def plot_remaining_expenses_pie_chart(self, top_percent=100):
        """
        Plots income and expense pie charts for the merged nodes in G.
        The nodes in G correspond to the keywords that we want to analyze.
        
        Parameters:
            top_percent (int): Percentage of top contributors to include in the pie chart (0 to 100).
        """
        
        # Ensure that 'G' exists and is already computed
        if not hasattr(self, 'G') or self.G is None:
            print("Graph G not found. Please run the network analysis first.")
            return

        # Filter the transactions dataframe based on the nodes in G (merged nodes)
        filtered_df = self.remainder_df[self.remainder_df[message_column].apply(
            lambda x: any(node in x.lower() for node in self.G.nodes))]

        # Initialize dictionaries to store the total amounts for income and expenses
        income_streams = defaultdict(float)
        expense_streams = defaultdict(float)

        # For each transaction, assign the net money stream to nodes
        for _, row in filtered_df.iterrows():
            message = row[message_column].lower()  # Lowercase message for easier matching
            income_amount = row['Beløp inn']  # Income amount
            expense_amount = row['Beløp ut']  # Expense amount

            # Loop through all nodes and check if they are in the message
            for node in self.G.nodes:
                # Check if the first word of the node name matches any part of the message
                node_name = node.split()[0].lower()
                
                if node_name in message:
                    if income_amount > 0:
                        income_streams[node] += income_amount
                    if expense_amount < 0:
                        expense_streams[node] += expense_amount

        # Convert negative expenses to positive for plotting
        expense_streams = {k: abs(v) for k, v in expense_streams.items()}

        # Sort income and expense streams by amount (in descending order)
        sorted_income_streams = dict(sorted(income_streams.items(), key=lambda item: item[1], reverse=True))
        sorted_expense_streams = dict(sorted(expense_streams.items(), key=lambda item: item[1], reverse=True))

        # Calculate the number of top contributors based on the percentage
        total_income = sum(sorted_income_streams.values())
        total_expenses = sum(sorted_expense_streams.values())

        # Determine how many items to include based on the percentage (top_percent)
        income_threshold = (top_percent / 100) * total_income
        expense_threshold = (top_percent / 100) * total_expenses

        # Filter the top contributors based on the calculated threshold
        filtered_income = {}
        cumulative_income = 0
        for node, amount in sorted_income_streams.items():
            cumulative_income += amount
            if cumulative_income <= income_threshold:
                filtered_income[node] = amount
            else:
                break

        filtered_expenses = {}
        cumulative_expenses = 0
        for node, amount in sorted_expense_streams.items():
            cumulative_expenses += amount
            if cumulative_expenses <= expense_threshold:
                filtered_expenses[node] = amount
            else:
                break

        # Plot income and expense pie charts
        plt.figure(figsize=(10, 5))

        # Plot income pie chart
        plt.subplot(1, 2, 1)  # First subplot (income)
        if filtered_income:
            plt.pie(filtered_income.values(), labels=filtered_income.keys(), autopct='%1.1f%%', startangle=90)
        else:
            plt.text(0.5, 0.5, 'No Income Data', horizontalalignment='center', verticalalignment='center')
        plt.title(f'Income Breakdown (Top {top_percent}% Contributors)')

        # Plot expense pie chart
        plt.subplot(1, 2, 2)  # Second subplot (expenses)
        if filtered_expenses:
            plt.pie(filtered_expenses.values(), labels=filtered_expenses.keys(), autopct='%1.1f%%', startangle=90)
        else:
            plt.text(0.5, 0.5, 'No Expense Data', horizontalalignment='center', verticalalignment='center')
        plt.title(f'Expense Breakdown (Top {top_percent}% Contributors)')

        plt.tight_layout()  # To adjust subplots and make sure they fit well
        plt.show()