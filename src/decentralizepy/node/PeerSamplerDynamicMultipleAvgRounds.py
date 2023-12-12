import logging
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
        if iteration != None:
            if iteration > self.iteration:
                logging.info(
                    "iteration, self.iteration: {}, {}".format(
                        iteration, self.iteration
                    )
                )
                assert averaging_round == 0  # Ensure we start a new iteration
                assert iteration == self.iteration + 1
                self.iteration = iteration
                self.averaging_round = 0
                # logging.debug(f"n_procs : {self.graph.n_procs}, degree :{self.graph_degree}")
                self.graphs.append([Regular(self.graph.n_procs, self.graph_degree)])
            elif iteration == self.iteration and averaging_round > self.averaging_round:
                logging.debug(
                    f"averaging_round, self.averaging_round: {averaging_round}, {self.averaging_round}"
                )
                assert averaging_round == self.averaging_round + 1, f"Expected to be averaging round {self.averaging_round + 1}, got {averaging_round}"
                self.averaging_round = averaging_round
                self.graphs[iteration].append(
                    Regular(self.graph.n_procs, self.graph_degree)
                )

            return self.graphs[iteration][averaging_round].neighbors(node)
        else:
            return self.graph.neighbors(node)

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
        graph_degree=2,
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
        args : optional
            Other arguments

        """

        self.iteration = -1
        self.averaging_round = 0

        self.total_averaging_rounds = averaging_rounds

        self.graphs = []
        self.graph_degree = graph_degree

        self.instantiate(
            rank = rank,
            machine_id=machine_id,
            mapping=mapping,
            graph=graph,
            config=config,
            iterations=iterations,
            log_dir=log_dir,
            log_level=log_level,
            *args,
        )

        self.run()

        logging.info("Peer Sampler exiting")
