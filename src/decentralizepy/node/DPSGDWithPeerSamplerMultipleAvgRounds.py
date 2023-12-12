import json
import logging
import math
import os
import sys
from collections import deque

import torch

from decentralizepy.graphs.Graph import Graph
from decentralizepy.mappings.Mapping import Mapping
from decentralizepy.node.DPSGDWithPeerSampler import DPSGDWithPeerSampler


class DPSGDWithPeerSamplerMultipleAvgRounds(DPSGDWithPeerSampler):
    """
    This class defines the node for DPSGD that performs multiple averaging rounds

    """

    def receive_DPSGD(self):
        sender, data = self.receive_channel("DPSGD")
        logging.info(
            f"Received Model from {sender} of iteration {data['iteration']} and round {data['averaging_round']}"
        )
        logging.debug(f"Complete message received: {data}")
        return sender, data

    def get_neighbors(self, node=None):
        logging.info("Requesting neighbors from the peer sampler.")
        self.communication.send(
            self.peer_sampler_uid,
            {
                "REQUEST_NEIGHBORS": self.uid,
                "iteration": self.iteration,
                "averaging_round": self.averaging_round,  # Problem: This increases the size of the data we must send when performing a single averaging round
                "CHANNEL": "SERVER_REQUEST",
            },
        )
        my_neighbors = self.receive_neighbors()
        logging.info(
            f"Neighbors for iteration {self.iteration} and round {self.averaging_round}: {my_neighbors}"
        )
        return my_neighbors

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
        weights_store_dir=".",
        log_level=logging.INFO,
        test_after=5,
        train_evaluate_after=1,
        reset_optimizer=1,
        peer_sampler_uid=-1,
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
        averaging_rounds : int
            Number of averaging rounds in a communication step. Default to 1.
        log_dir : str
            Logging directory
        weights_store_dir : str
            Directory in which to store model weights
        log_level : logging.Level
            One of DEBUG, INFO, WARNING, ERROR, CRITICAL
        test_after : int
            Number of iterations after which the test loss and accuracy are calculated
        train_evaluate_after : int
            Number of iterations after which the train loss is calculated
        reset_optimizer : int
            1 if optimizer should be reset every communication round, else 0
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
            weights_store_dir=weights_store_dir,
            log_level=log_level,
            test_after=test_after,
            train_evaluate_after=train_evaluate_after,
            reset_optimizer=reset_optimizer,
            peer_sampler_uid=peer_sampler_uid,
            *args,
        )

        # total_threads = os.cpu_count()
        # self.threads_per_proc = max(
        #     math.floor(total_threads / mapping.procs_per_machine), 1
        # )
        # torch.set_num_threads(self.threads_per_proc)
        # torch.set_num_interop_threads(1)
        # self.instantiate(
        #     rank=rank,
        #     machine_id=machine_id,
        #     mapping=mapping,
        #     graph=graph,
        #     config=config,
        #     iterations=iterations,
        #     log_dir=log_dir,
        #     weights_store_dir=weights_store_dir,
        #     log_level=log_level,
        #     test_after=test_after,
        #     train_evaluate_after=train_evaluate_after,
        #     reset_optimizer=reset_optimizer,
        #     *args,
        # )
        # logging.info(f"RANK : {rank}, MACHINE ID : {machine_id}")

        # self.message_queue["PEERS"] = deque()

        # self.peer_sampler_uid = peer_sampler_uid
        # self.connect_neighbor(self.peer_sampler_uid)
        # self.wait_for_hello(self.peer_sampler_uid)

        # self.run()

    def run(self):
        """
        Start the decentralized learning

        """
        self.testset = self.dataset.get_testset()
        rounds_to_test = self.test_after
        rounds_to_train_evaluate = self.train_evaluate_after
        global_epoch = 1
        change = 1

        for iteration in range(self.iterations):
            logging.info(f"Starting training iteration: {iteration}")
            rounds_to_train_evaluate -= 1
            rounds_to_test -= 1

            self.iteration = iteration
            self.trainer.train(self.dataset)

            # The following code does not work because TCP sockets are supposed to be long lived.
            # for neighbor in self.my_neighbors:
            #     if neighbor not in new_neighbors:
            #         logging.info("Removing neighbor {}".format(neighbor))
            #         if neighbor in self.peer_deques:
            #             assert len(self.peer_deques[neighbor]) == 0
            #             del self.peer_deques[neighbor]
            #         self.communication.destroy_connection(neighbor, linger = 10000)
            #         self.barrier.remove(neighbor)

            for self.averaging_round in range(self.averaging_rounds):
                logging.info(f"Starting averaging round: {self.averaging_round}")
                new_neighbors = self.get_neighbors()
                self.my_neighbors = new_neighbors

                self.connect_neighbors()

                logging.info("Connected to all neighbors")

                # self.instantiate_peer_deques()

                self.sharing.send_all(
                    self.my_neighbors, averaging_round=self.averaging_round
                )

                while not self.received_from_all():
                    sender, data = self.receive_DPSGD()
                    if sender not in self.peer_deques:
                        self.peer_deques[sender] = deque()
                    self.peer_deques[sender].append(data)

                averaging_deque = dict()
                for neighbor in self.my_neighbors:
                    averaging_deque[neighbor] = self.peer_deques[neighbor]

                self.sharing._averaging(averaging_deque)

            if self.reset_optimizer:
                self.optimizer = self.optimizer_class(
                    self.model.parameters(), **self.optimizer_params
                )  # Reset optimizer state
                self.trainer.reset_optimizer(self.optimizer)

            if iteration:
                with open(
                    os.path.join(self.log_dir, "{}_results.json".format(self.rank)),
                    "r",
                ) as inf:
                    results_dict = json.load(inf)
            else:
                results_dict = {
                    "train_loss": {},
                    "test_loss": {},
                    "test_acc": {},
                    "total_bytes": {},
                    "total_meta": {},
                    "total_data_per_n": {},
                }

            results_dict["total_bytes"][iteration + 1] = self.communication.total_bytes

            if hasattr(self.communication, "total_meta"):
                results_dict["total_meta"][
                    iteration + 1
                ] = self.communication.total_meta
            if hasattr(self.communication, "total_data"):
                results_dict["total_data_per_n"][
                    iteration + 1
                ] = self.communication.total_data

            if rounds_to_train_evaluate == 0:
                logging.info("Evaluating on train set.")
                rounds_to_train_evaluate = self.train_evaluate_after * change
                loss_after_sharing = self.trainer.eval_loss(self.dataset)
                results_dict["train_loss"][iteration + 1] = loss_after_sharing
                self.save_plot(
                    results_dict["train_loss"],
                    "train_loss",
                    "Training Loss",
                    "Communication Rounds",
                    os.path.join(self.log_dir, "{}_train_loss.png".format(self.rank)),
                )

            if self.dataset.__testing__ and rounds_to_test == 0:
                rounds_to_test = self.test_after * change
                logging.info("Evaluating on test set.")
                ta, tl = self.dataset.test(self.model, self.loss)
                results_dict["test_acc"][iteration + 1] = ta
                results_dict["test_loss"][iteration + 1] = tl

                if global_epoch == 49:
                    change *= 2

                global_epoch += change

            with open(
                os.path.join(self.log_dir, "{}_results.json".format(self.rank)), "w"
            ) as of:
                json.dump(results_dict, of)
        if self.model.shared_parameters_counter is not None:
            logging.info("Saving the shared parameter counts")
            with open(
                os.path.join(
                    self.log_dir, "{}_shared_parameters.json".format(self.rank)
                ),
                "w",
            ) as of:
                json.dump(self.model.shared_parameters_counter.numpy().tolist(), of)
        self.disconnect_neighbors()
        logging.info("Storing final weight")
        self.model.dump_weights(self.weights_store_dir, self.uid, iteration)
        logging.info("All neighbors disconnected. Process complete!")
