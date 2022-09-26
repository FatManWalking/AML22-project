import numpy as np
import pandas as pd
import networkx as nx
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
from pymer4.models import Lmer  # just import the linear mixed models class
import scipy.stats as stats

# Load the data from ../analysis/ENZYMES/graphs_{i}.npy
# where i is the index of the graph
def load_data(i):
    return np.load(
        "analysis/emp-hgpsl/ENZYMES/graphs_{}.npy".format(i), allow_pickle=True
    )


def get_cna_metrics():

    # list of graph metrics for each dataset
    all_graph_metrics = []
    datasets = [load_data(i) for i in range(5)]

    for dataset in datasets:
        graph_metrics = []
        for graph in dataset:

            graph_metrics.append(
                [
                    graph.number_of_nodes(),
                    nx.graph_number_of_cliques(graph),
                    max(dict(graph.degree).values()),
                    min(dict(graph.degree).values()),
                    np.mean(list(dict(graph.degree).values())),
                    # nx.degree_assortativity_coefficient(graph),
                    nx.density(graph),
                    np.mean(list(dict(graph.degree).values())),
                    nx.average_clustering(graph),
                    graph.graph["graph_id"],
                    graph.graph["prediction"],
                    graph.graph["label"],
                    graph.graph["confidence"],
                    graph.graph["correct"],
                    graph.graph["loss"],
                    graph.graph["replication"],
                ]
            )
        all_graph_metrics.append(graph_metrics)

    return all_graph_metrics


def rename_columns(df):
    # maps the index of graph_metrics to the metric name (needed to label automatically in visualization)
    index_to_metric = {
        0: "Number_of_Vertices",
        1: "Number_of_Cliques",
        2: "Maximum_Degree",
        3: "Minimum_Degree",
        4: "Average_Degree",
        5: "Density",
        6: "Average_Neighbor_Degree",
        7: "Average_Clustering_Coefficient",
        8: "Graph_ID",
        9: "Prediction",
        10: "Label",
        11: "Confidence",
        12: "Correct",
        13: "Loss",
        14: "Replication",
    }

    df = df.rename(columns=index_to_metric)
    return df


# build a dataframe for each dataset
def build_dataframe(all_graph_metrics):
    df_list = []
    for graph_metrics in all_graph_metrics:
        df = pd.DataFrame(graph_metrics)
        df = rename_columns(df)
        df_list.append(df)
    return df_list


if __name__ == "__main__":

    df_list = build_dataframe(get_cna_metrics())
    architecture = "emp-hgpsl"
    df = pd.concat(df_list, ignore_index=True)
    df["system"] = architecture

    eval_data = df.astype(
        {
            "Graph_ID": "category",
            "Correct": "category",
            "Prediction": "category",
            "Label": "category",
            "Replication": "category",
            "system": "category",
        }
    )

    # filter eval_data to only include "Replication" 1, 2, 3
    # eval_data = eval_data[eval_data["Replication"].isin(["1", "2", "3"])]

    differentMeans_model = Lmer(
        formula="Loss ~ Replication + (1 | Graph_ID)", data=eval_data
    )

    differentMeans_model.fit(
        factors={"Replication": ["1", "2", "3", "4", "5"]}, REML=False, summarize=False
    )
