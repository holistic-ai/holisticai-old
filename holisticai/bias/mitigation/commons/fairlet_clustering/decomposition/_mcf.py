import random
import time
from math import gcd

import networkx as nx
import numpy as np

from holisticai.bias.mitigation.commons.fairlet_clustering._utils import distance

from ._base import DecompositionMixin


class MCFFairletDecomposition(DecompositionMixin):
    """
    Computes the optimized version of fairlet decomposition using minimum-cost flow.
    """

    def __init__(self, t, distance_threshold):
        """
        blues (list) : Index of the points corresponding to first class
        reds (list) : Index of the points corresponding to second class
        t (int) : (1, t) is the fairness ratio to be enforced
        distance_threshold (int) : Value to be used for pushing cost to infinity
        data (list) : Contains actual data points
        """
        self.t = t
        self.distance_threshold = distance_threshold

        # Initializing the Graph
        self.G = nx.DiGraph()

    def compute_distances(self):
        """
        Compute distances between every pair of blue and red nodes.
        """
        random.seed(42)
        random.shuffle(self.blues)
        random.shuffle(self.reds)

        self.distances = {}
        for idx, i in enumerate(self.blues):
            for idx2, j in enumerate(self.reds):
                self.distances["B_%d_R_%d" % (idx + 1, idx2 + 1)] = distance(
                    self.data[i], self.data[j]
                )

    def build_graph(self, plot_graph=False, weight_limit=10000000):
        """
        Builds the graph i.e. nodes and edges.

        Args:
                plot_graph (bool) : Indicates whether the graph needs to be plotted
                weight_limit (int) : Big value to be used in place of infinity for cost definition
        """

        self.G.add_node(
            "beta",
            pos=(0, 4 + (1 + max(self.blue_nodes, self.red_nodes)) / 2),
            demand=(-1 * self.red_nodes),
        )
        self.G.add_node(
            "ro",
            pos=(5, 4 + (1 + max(self.blue_nodes, self.red_nodes)) / 2),
            demand=(self.blue_nodes),
        )
        self.G.add_edge(
            "beta", "ro", weight=0, capacity=min(self.blue_nodes, self.red_nodes)
        )

        for i in range(self.blue_nodes):
            self.G.add_node("B%d" % (i + 1), pos=(1, i + 1), demand=-1)
            self.G.add_edge("beta", "B%d" % (i + 1), weight=0, capacity=self.t - 1)
        for i in range(self.red_nodes):
            self.G.add_node("R%d" % (i + 1), pos=(4, i + 1), demand=1)
            self.G.add_edge("R%d" % (i + 1), "ro", weight=0, capacity=self.t - 1)

        # Latent nodes
        for i in range(self.blue_nodes):
            for j in range(self.t):
                position = (i + 1) + ((i + 1 - i) / self.t) * j
                self.G.add_node("B%d_%d" % (i + 1, j + 1), pos=(2, position), demand=0)
                self.G.add_edge(
                    "B%d" % (i + 1), "B%d_%d" % (i + 1, j + 1), weight=0, capacity=1
                )
        for i in range(self.red_nodes):
            for j in range(self.t):
                position = (i + 1) + ((i + 1 - i) / self.t) * j
                self.G.add_node("R%d_%d" % (i + 1, j + 1), pos=(3, position), demand=0)
                self.G.add_edge(
                    "R%d_%d" % (i + 1, j + 1), "R%d" % (i + 1), weight=0, capacity=1
                )

        # Adding edges between latent nodes
        for i in range(self.blue_nodes):
            for j in range(self.t):
                for k in range(self.red_nodes):
                    for l in range(self.t):
                        dist = self.distances["B_%d_R_%d" % (i + 1, k + 1)]
                        if dist <= self.distance_threshold:
                            self.G.add_edge(
                                "B%d_%d" % (i + 1, j + 1),
                                "R%d_%d" % (k + 1, l + 1),
                                weight=1,
                                capacity=1,
                            )
                        else:
                            self.G.add_edge(
                                "B%d_%d" % (i + 1, j + 1),
                                "R%d_%d" % (k + 1, l + 1),
                                weight=weight_limit,
                                capacity=1,
                            )

        if plot_graph:
            if self.blue_nodes > 10:
                print("Graph can't be plotted because the blue nodes exceed 10.")
            else:
                plt.figure(figsize=(10, 8))
                pos = {
                    n: (x, y)
                    for (n, (x, y)) in nx.get_node_attributes(self.G, "pos").items()
                }
                nx.draw_networkx_nodes(self.G, pos, node_size=1000, alpha=0.5)
                nx.draw_networkx_labels(self.G, pos, font_size=11)
                nx.draw_networkx_edges(self.G, pos)
                plt.show()

    def fit_transform(self, data, group_a, group_b):
        """
        Calls the network simplex to run the MCF algorithm.
        Computes the fairlets and fairlet centers.

        Returns:
                fairlets (list)
                fairlet_centers (list)
                costs (list)
        """
        blues = list(np.where(group_a)[0])
        reds = list(np.where(group_b)[0])
        self.blues = blues
        self.blue_nodes = len(blues)
        self.reds = reds
        self.red_nodes = len(reds)
        assert self.blue_nodes >= self.red_nodes
        self.data = data

        self.compute_distances()
        self.build_graph(plot_graph=True)

        start_time = time.time()
        flow_cost, flow_dict = nx.network_simplex(self.G)
        print(
            "Time taken to compute MCF solution - %.3f seconds."
            % (time.time() - start_time)
        )

        fairlets = {}
        # Assumes mapping from blue nodes to the red nodes
        for i in flow_dict.keys():
            if "B" in i and "_" in i:
                if sum(flow_dict[i].values()) == 1:
                    for j in flow_dict[i].keys():
                        if flow_dict[i][j] == 1:
                            if j.split("_")[0] not in fairlets:
                                fairlets[j.split("_")[0]] = [i.split("_")[0]]
                            else:
                                fairlets[j.split("_")[0]].append(i.split("_")[0])

        fairlets = [([a] + b) for a, b in fairlets.items()]

        fairlets2 = []
        for i in fairlets:
            curr_fairlet = []
            for j in i:
                if "R" in j:
                    d = self.reds
                else:
                    d = self.blues
                curr_fairlet.append(d[int(j[1:]) - 1])
            fairlets2.append(curr_fairlet)
        fairlets = fairlets2
        del fairlets2

        # Choosing fairlet centers
        fairlet_centers = []
        fairlet_costs = []

        for f in fairlets:
            cost_list = [
                (i, max([distance(self.data[i], self.data[j]) for j in f])) for i in f
            ]
            cost_list = sorted(cost_list, key=lambda x: x[1], reverse=False)
            center, cost = cost_list[0][0], cost_list[0][1]
            fairlet_centers.append(center)
            fairlet_costs.append(cost)

        print("%d fairlets have been identified." % (len(fairlet_centers)))
        assert len(fairlets) == len(fairlet_centers)
        assert len(fairlet_centers) == len(fairlet_costs)

        return fairlets, fairlet_centers, fairlet_costs
