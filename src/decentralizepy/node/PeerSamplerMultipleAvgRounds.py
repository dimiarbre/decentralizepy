import importlib
import logging
import os
from collections import deque

from decentralizepy import utils
from decentralizepy.graphs.Graph import Graph
from decentralizepy.mappings.Mapping import Mapping
from decentralizepy.node.PeerSampler import PeerSampler


class PeerSamplerMultipleAvgRounds(PeerSampler):
    """
    This class defines the peer sampling service

    """


    def init_log(self, log_dir, log_level, force=True):
        """
        Instantiate Logging.

        Parameters
        ----------
        log_dir : str
            Logging directory
        rank : rank : int
            Rank of process local to the machine
        log_level : logging.Level
            One of DEBUG, INFO, WARNING, ERROR, CRITICAL
        force : bool
            Argument to logging.basicConfig()

        """
        log_file = os.path.join(log_dir, "PeerSamplerMultipleAvgRounds.log")
        logging.basicConfig(
            filename=log_file,
            format="[%(asctime)s][%(module)s][%(levelname)s] %(message)s",
            level=log_level,
            force=force,
        )

    def get_neighbors(self, node, iteration=None, averaging_round=0):
        return self.graph.neighbors(node)

    def run(self):
        """
        Start the peer-sampling service.

        """
        while len(self.barrier) > 0:
            sender, data = self.receive_server_request()
            if "BYE" in data:
                logging.debug("Received {} from {}".format("BYE", sender))
                self.barrier.remove(sender)

            elif "REQUEST_NEIGHBORS" in data:
                logging.debug("Received {} from {}".format("Request", sender))
                if "iteration" in data:
                    if "averaging_round" in data:
                        resp = {
                            "NEIGHBORS": self.get_neighbors(
                                sender, data["iteration"], data["averaging_round"]
                            ),
                            "CHANNEL": "PEERS",
                        }
                    else:
                        resp = {
                            "NEIGHBORS": self.get_neighbors(sender, data["iteration"]),
                            "CHANNEL": "PEERS",
                        }

                else:
                    resp = {"NEIGHBORS": self.get_neighbors(sender), "CHANNEL": "PEERS"}
                self.communication.send(sender, resp)

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
        *args
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
        averaging_rounds : int
            Number of averaging rounds in an iteration (typically between two gradient descent).
        log_dir : str
            Logging directory
        log_level : logging.Level
            One of DEBUG, INFO, WARNING, ERROR, CRITICAL
        args : optional
            Other arguments

        """
        self.averaging_rounds = averaging_rounds

        super().__init__(
            rank=rank,
            machine_id=machine_id,
            mapping=mapping,
            graph=graph,
            config=config,
            iterations=iterations,
            log_dir=log_dir,
            log_level=log_level,
            *args
        )
