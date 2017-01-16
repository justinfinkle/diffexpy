__author__ = 'Justin Finkle'
__email__ = 'jfinkle@u.northwestern.edu'

import sys
import numpy as np
import pandas as pd

def save_gdf(df, path):
    # Headers
    node_header = 'nodedef>name VARCHAR\n'
    columns = df.columns
    headers = ''.join([', '+col+' DOUBLE' for col in columns if (col!='Parent' and col!='Child')])
    headers = headers.replace('directed DOUBLE', 'directed BOOLEAN')
    edge_header = 'edgedef>node1 VARCHAR, node2 VARCHAR'+headers+'\n'

    # Build node set
    nodes = list(set(np.hstack((df.Parent, df.Child))))
    print "Saving file with %i nodes and %i edges" %(len(nodes), len(df))
    write_file = open(path, 'w')

    # Write nodes
    write_file.write(node_header)
    for node in nodes:
        write_file.write(str(node)+"\n")

    # Write edges
    write_file.write(edge_header)
    for row in df.values.astype(str):
        write_file.write(",".join(row)+"\n")

    write_file.close()

def load_link_list(path):
    df = pd.read_csv(path)
    return df

if __name__ == '__main__':
    unfiltered = "network_inference/20160617_prefiltered_dionesus_2pc_log_fc_intersection_link_list.csv"
    df = load_link_list(unfiltered)
    df.drop(df.columns.values[0], axis=1, inplace=True)
    p_thresh = 0.01
    filtered_df = df[(df['VIP_pval'] <= p_thresh)].copy()
    print df.shape
    print filtered_df.shape
    filtered_df.insert(2, 'directed', np.ones((len(filtered_df), 1), dtype=bool))
    df.to_csv("network_inference/20160616_dionesus_2pc_log_fc_intersection_all_link_list_filtered.csv")
    save_gdf(filtered_df, "network_inference/20160617_dionesus_2pc_log_fc_intersection_all_link_list_filtered.gdf")
    sys.exit()


    dionesus_1pc = "network_inference/20150928_dionesus_1pc_log_fc_intersection_all_link_list.csv"
    dionesus_2pc = "network_inference/20150912_dionesus_2pc_log_fc_intersection_all_link_list.csv"

    df_1pc = load_link_list(dionesus_1pc)
    df_2pc = load_link_list(dionesus_2pc)
    df_1pc.drop(df_1pc.columns.values[0], axis=1, inplace=True)
    df_2pc.drop(df_2pc.columns.values[0], axis=1, inplace=True)

    # Only keep edges with a VIP pval that meet the threshold
    p_thresh = 0.05
    filtered_1pc = df_1pc[df_1pc['VIP_pval']<=p_thresh].copy()
    filtered_2pc = df_2pc[df_2pc['VIP_pval']<=p_thresh].copy()
    filtered_1pc.index = zip(filtered_1pc['Parent'].values, filtered_1pc['Child'].values)
    filtered_2pc.index = zip(filtered_2pc['Parent'].values, filtered_2pc['Child'].values)

    # Calculate unique edges
    edges_1pc = set(filtered_1pc.index.values)
    edges_2pc = set(filtered_2pc.index.values)

    edge_union = edges_1pc.union(edges_2pc)                         # Edge in either
    edge_intersection = list(edges_1pc.intersection(edges_2pc))     # Edge in both
    unique_1pc = list(edges_1pc.difference(edges_2pc))              # Edge only in 1pc
    unique_2pc = list(edges_2pc.difference(edges_1pc))              # Edge only in 2pc

    # Make unique dataframes
    unique_1pc_df = filtered_1pc.loc[unique_1pc].copy().sort(['VIP'], ascending=False)
    unique_2pc_df = filtered_2pc.loc[unique_2pc].copy().sort(['VIP'], ascending=False)
    joint_df = filtered_2pc.loc[edge_intersection].copy().sort(['VIP'], ascending=False)

    #unique_1pc_df.to_csv("network_inference/20150929_dionesus_unique_1pc_log_fc_intersection_all_link_list.csv")
    #unique_2pc_df.to_csv("network_inference/20150929_dionesus_unique_2pc_log_fc_intersection_all_link_list.csv")
    #joint_df.to_csv("network_inference/20150929_dionesus_both_1pc_2pc_log_fc_intersection_all_link_list.csv")
    save_gdf(unique_2pc_df, "20150929_dionesus_unique_2pc_log_fc_intersection_all_link_list.gdf")