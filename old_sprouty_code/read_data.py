__author__ = 'Justin Finkle'
__email__ = 'jfinkle@u.northwestern.edu'

"""
This is a short script to parse the rawest of the raw data
"""

import sys
import pandas as pd
import numpy as np
import itertools

def make_dataframe(file_path):
    # Read the dataframe
    df = pd.read_csv(file_path, sep='\t', skiprows=7, header=1)
    gene_names = df['TargetID']

    # Find the columns that are the average signal and keep those
    data_df = df[[col for col in df.columns if "AVG_Signal" in col]]
    data_df.columns = replace_column_labels(data_df.columns, 'Null', 'KO')
    data_df.columns = replace_column_labels(data_df.columns, 'AVG_Signal-BN-', '')
    #print(data_df.head(5))
    column_df = pd.DataFrame([col.split('-') for col in data_df.columns], index=data_df.columns)

    # There is one extra timeopint for Replicate A. Correct the dataframe
    column_df.loc[:,2] = column_df.loc[:,2].replace('1', '0A')
    column_df.drop(3,1, inplace=True)
    column_df.columns = ['Condition', 'RunNum', 'Time_Replicate']
    # Put the gene names back in the dataframe
    data_df.insert(0, 'TargetID', gene_names.copy())

    return data_df, column_df

def prepare_dataframe(data_df, column_df):
    # Split up the times and replicates

    times,replicates = zip(*[(int(col[:-1]), col[-1]) for col in column_df['Time_Replicate']])
    time_set = list(set(times))
    time_set.sort()
    column_df['Times'] = times
    column_df['Replicate'] = replicates

    # NOTE: This part is manual
    # Remove the two weird columns
    data_df.drop(['WT-1-1-0A', 'KO-16-1-0A'], 1, inplace=True)
    column_df.drop(['WT-1-1-0A', 'KO-16-1-0A'], 0, inplace=True)
    #print(np.array_equal(data_df.columns.values[1:], column_df.index.values))

    # Slice the data_df columns in the order we want (every third)
    n_columns = len(data_df.columns)
    idx = list(range(1, n_columns, 3))+list(range(2, n_columns, 3))+list(range(3, n_columns, 3))

    df = data_df.iloc[:, idx]
    df.insert(0, 'Gene', data_df['TargetID'])
    df = df.append([column_df['Times'], column_df['Replicate'], column_df['Condition']])
    df.loc['Times', 'Gene'] = 'Time'
    df.loc['Replicate', 'Gene'] = 'Replicate'
    df.loc['Condition', 'Gene'] = 'Condition'
    return df.T

def save_to_csv(data_df, column_df, save_loc):

    # Store the replicates and conditions as separate files
    conditions = list(set(column_df.loc[:,2]))
    times,replicates = zip(*[(int(col[:-1]), col[-1]) for col in column_df.loc[:,4]])
    time_set = list(set(times))
    replicate_set = list(set(replicates))
    column_df['Times'] = times
    column_df['Replicate'] = replicates
    #print(column_data)

    for condition in conditions:
        for replicate in replicate_set:
            save_string = save_loc + condition + "_rep" + replicate + "_gene_level.csv"
            if condition == 'Null':
                save_string = save_string.replace('Null', 'KO')

            column_select = column_df.index[(column_df[2]==condition)
                                              & (column_df['Replicate'] == replicate)].values
            save_df = data_df.loc[:, column_select]
            save_df.insert(0, 'TargetID', data_df['TargetID'].copy())
            save_df.to_csv(save_string, index=False)

def save_for_sam(data_df, column_df, save_loc):
    """
    SAM = Significant analysis of Microarray

    Requires data to be in specific format for analysis. This will be for the Two Class Paired Time Course

    :param data_df:
    :param column_df:
    :param save_loc:
    :return:
    """

    # Store the replicates and conditions as separate files
    conditions = list(set(column_df['Condition']))

    times,replicates = zip(*[(int(col[:-1]), col[-1]) for col in column_df['Time_Replicate']])
    time_set = list(set(times))
    time_set.sort()
    replicate_set = list(set(replicates))
    column_df['Times'] = times
    column_df['Replicate'] = replicates
    labels = make_sam_column_labels(conditions, time_set, len(replicate_set))
    labels.insert(0,'')

    # NOTE: This part is manual
    # Remove the two weird columns
    data_df.drop(['WT-1-1-0A', 'KO-16-1-0A'], 1, inplace=True)

    # Slice the data_df columns in the order we want (every third)
    n_columns = len(data_df.columns)
    idx = list(range(1, n_columns, 3))+list(range(2, n_columns, 3))+list(range(3, n_columns, 3))

    save_df = data_df.iloc[:, idx]
    save_df.insert(0, '', data_df['TargetID'])
    print(save_df.head(5))
    sys.exit()
    save_df.columns = labels


    save_df.to_csv(save_loc+"sam_data.csv")
    return

def make_sam_column_labels(conditions, times, n_replicates=1):
    """
    Make the column labels.

    For Two Class Paired Time course they must be in the format: [Class]Time[Int](Start/End), where class is 1 or 2
    examples: 1Time0Start, 1Time2, 1Time3End, 2Time0Start, 2Time2, 2Time3End

    :param conditions:
    :param times:
    :param n_replicates:
    :return:
    """

    # NOTE : I think replicates can just have additional columns with the same labels

    # Make sure times are sorted low to high
    times.sort()
    min_time = min(times)
    max_time = max(times)

    # Reassign the conditions to SAM format of integers
    sam_conditions = range(1, len(conditions)+1)

    column_labels = [str(condition) + 'Time' + str(time) + 'Start' if time == min_time
                     else str(condition) + 'Time' + str(time) + 'End' if time == max_time
                     else str(condition) + 'Time' + str(time)
                     for condition in sam_conditions for time in times]*n_replicates

    return column_labels

def replace_column_labels(labels, to_replace, replace_with):
    new_labels = [label.replace(to_replace, replace_with) for label in labels]
    return new_labels


if __name__ == '__main__':
    file_path = "data/raw_data/Licht-BN-Jul23-11-M8v2_GeneSpring_GeneLevel_QuantileNormalized_BGSubtracted.txt"
    save_loc = "data/raw_data/"

    # Get the raw data frames
    data_df, column_df = make_dataframe(file_path)

    clean_df = prepare_dataframe(data_df, column_df)
    clean_df.to_pickle(save_loc+"all_data.pickle")
    filtered_df = clean_df[(clean_df['Condition']=='KO')|(clean_df['Condition']=='Condition')]
    print(filtered_df.iloc[:,:-2])
    #save_to_csv(data_df, column_df, save_loc)
    #save_for_sam(data_df, column_df, "data/")
