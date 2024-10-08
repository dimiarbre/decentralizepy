import logging
import os
from collections import deque

from decentralizepy.graphs.Graph import Graph
from decentralizepy.graphs.Regular import Regular
from decentralizepy.mappings.Mapping import Mapping
from decentralizepy.node.PeerSamplerDynamic import PeerSamplerDynamic


class PeerSamplerDynamicMultipleAvgRounds(PeerSamplerDynamic):
    """
    This class defines the peer sampling service

    """

    def get_neighbors(self, node, iteration=None, averaging_round=0):
        # logging.debug(f"Node : {node}, {iteration}, {averaging_round} ")
        if (
            not self.dynamic_avgrounds
        ):  # We only want the same graph, super() already handles the job, but not the API
            return super().get_neighbors(node, iteration=iteration)
        if iteration is not None:
            current_seed = (
                self.random_seed * 100000 + 10000000 * iteration + averaging_round
            )
            if iteration > self.iteration:
                # We start a new iteration
                logging.debug(
                    "iteration, self.iteration: {}, {}".format(
                        iteration, self.iteration
                    )
                )
                assert averaging_round == 0  # Ensure we start a new iteration
                assert iteration == self.iteration + 1
                self.iteration = iteration
                self.averaging_round = 0
                # logging.debug(f"n_procs : {self.graph.n_procs}, degree :{self.graph_degree}")
                self.graphs_lists.append(
                    [Regular(self.graph.n_procs, self.graph_degree, seed=current_seed)]
                )
            elif iteration == self.iteration and averaging_round > self.averaging_round:
                # We start a new averaging round inside the current iteration.
                logging.debug(
                    f"averaging_round, self.averaging_round: {averaging_round}, {self.averaging_round}"
                )
                assert (
                    averaging_round == self.averaging_round + 1
                ), f"Expected to be averaging round {self.averaging_round + 1}, got {averaging_round}"
                self.averaging_round = averaging_round
                self.graphs_lists[iteration].append(
                    Regular(self.graph.n_procs, self.graph_degree, seed=current_seed)
                )

            return self.graphs_lists[iteration][averaging_round].neighbors(node)
        else:
            return self.graph.neighbors(node)

    def handle_request_neighors(self, data, sender):
        """Handles a request to the Peer Sampler

        Args:
            data (dic): The request
            sender (int): The sender of the request

        Returns:
            dic: The formatted response to the neighbors request.
        """
        if "iteration" in data and "averaging_round" in data:
            # This will work only with PeerSampleMultipleAvgRound subclasses
            resp = {
                "NEIGHBORS": self.get_neighbors(
                    sender, data["iteration"], data["averaging_round"]
                ),
                "CHANNEL": "PEERS",
            }
            return resp
        else:
            return super().handle_request_neighors(data, sender)

    def log_graphs(self):
        logging.info("Saving graphs:")
        graph_log_dir = f"{self.log_dir}/graphs/"
        os.mkdir(graph_log_dir)
        if not self.dynamic_avgrounds:  # If we have dynamicity only between GD
            for iteration, graph in enumerate(self.graphs):
                graph.write_graph_to_file(f"{graph_log_dir}{iteration}")
        else:  # When we randomize each graph between averaging rounds.
            for iteration, iteration_graph_list in enumerate(self.graphs_lists):
                for averaging_round, graph in enumerate(iteration_graph_list):
                    graph.write_graph_to_file(
                        f"{graph_log_dir}{iteration}_{averaging_round}"
                    )
        logging.info("Saved graphs")

    def __init__(
        self,
        rank: int,
        machine_id: int,
        mapping: Mapping,
        graph: Graph,
        config,
        iterations=1,
        averaging_rounds=1,
        log_dir=".",
        log_level=logging.INFO,
        dynamic_avgrounds=False,
        *args,
    ):
        """
        Constructor

        Parameters
        ----------
        rank : int
            Rank of process local to the machine
        machine_id : int
            Machine ID on which the process in running
        mapping : decentralizepy.mappings
            The object containing the mapping rank <--> uid
        graph : decentralizepy.graphs
            The object containing the global graph
        config : dict
            A dictionary of configurations. Must contain the following:
            [DATASET]
                dataset_package
                dataset_class
                model_class
            [OPTIMIZER_PARAMS]
                optimizer_package
                optimizer_class
            [TRAIN_PARAMS]
                training_package = decentralizepy.training.Training
                training_class = Training
                epochs_per_round = 25
                batch_size = 64
        iterations : int
            Number of iterations (communication steps) for which the model should be trained
        log_dir : str
            Logging directory
        log_level : logging.Level
            One of DEBUG, INFO, WARNING, ERROR, CRITICAL
        dynamic_avgrounds : bool, default False
            If true, randomize between each averaging round, if False randomize the graph only
            between gradient descent steps.
        args : optional
            Other arguments

        """

        self.iteration = -1
        self.averaging_round = 0

        self.dynamic_avgrounds = dynamic_avgrounds
        logging.debug("dynamic_avgrounds set to %s", self.dynamic_avgrounds)

        self.total_averaging_rounds = averaging_rounds

        self.graphs: list[Graph] = []
        self.graphs_lists: list[list[Graph]] = []

        node_configs = config["NODE"]
        self.graph_degree = node_configs["graph_degree"]

        self.instantiate(
            rank=rank,
            machine_id=machine_id,
            mapping=mapping,
            graph=graph,
            config=config,
            iterations=iterations,
            log_dir=log_dir,
            log_level=log_level,
            # *args,
        )

        self.run()

        self.log_graphs()

        logging.info("Peer Sampler exiting")
