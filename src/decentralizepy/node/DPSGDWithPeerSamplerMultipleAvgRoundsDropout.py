import json
import logging
import math
import os
import random
import time
from collections import deque
from logging import INFO

import torch

from decentralizepy.graphs.Graph import Graph
from decentralizepy.mappings.Mapping import Mapping
from decentralizepy.node.DPSGDWithPeerSamplerMultipleAvgRounds import (
    DPSGDWithPeerSamplerMultipleAvgRounds,
)
from decentralizepy.sharing.SharingAsymmetric import SharingAsymmetric
from decentralizepy.utils import error_logging_wrapper


class DPSGDWithPeerSamplerMultipleAvgRoundsDropout(
    DPSGDWithPeerSamplerMultipleAvgRounds
):
    def participate(self):
        """
        Participate in the current round

        """

        if self.dropped_last_round:
            to_crash = self.dropout_rng.random() < (
                self.dropout_rate + self.dropout_correlation * (1 - self.dropout_rate)
            )
        else:
            to_crash = self.dropout_rng.random() < self.dropout_rate

        return not to_crash

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
        dropout_rate=0.1,
        dropout_correlation=0.1,
        should_run=True,
        *args,
    ):
        super().__init__(
            rank,
            machine_id,
            mapping,
            graph,
            config,
            iterations,
            averaging_rounds,
            log_dir,
            weights_store_dir,
            log_level,
            test_after,
            train_evaluate_after,
            reset_optimizer,
            peer_sampler_uid,
            should_run=False,
            *args,
        )
        self.dropout_rng = random.Random()

        self.dropout_rng.seed(self.dataset.random_seed * 125 + self.uid)
        self.dropout_rate = dropout_rate
        self.dropout_correlation = dropout_correlation
        self.dropped_last_round = False

        assert isinstance(self.sharing, SharingAsymmetric)
        self.sharing: SharingAsymmetric

        if should_run:
            self.run()

    @error_logging_wrapper
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
            self.sharing.training_iteration = iteration
            to_participate = self.participate()

            if to_participate:
                self.trainer.train(self.dataset)
            else:
                logging.debug(
                    "Node %s not participating at iteration %s.",
                    self.uid,
                    self.iteration,
                )

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

                self.dropped_last_round = to_participate

                if to_participate:
                    self.sharing.send_all(
                        self.my_neighbors, averaging_round=self.averaging_round
                    )
                else:
                    self.sharing.send_dropped_out(
                        self.my_neighbors, averaging_round=self.averaging_round
                    )

                while not self.received_from_all():
                    sender, data = self.receive_DPSGD()
                    if sender not in self.peer_deques:
                        self.peer_deques[sender] = deque()
                    self.peer_deques[sender].append(data)

                # # Filter by removing data corresponding to dropped out nodes
                # if "DROPPED_OUT" in data:
                #     # Ignore messages from dropped out nodes.
                #     logging.debug("Received DROPOUT signal from %s", sender)
                #     continue

                # Only participate if you are not dropped out.
                if to_participate:
                    logging.debug("Starting averaging")
                    averaging_deque = dict()
                    for neighbor in self.my_neighbors:
                        averaging_deque[neighbor] = self.peer_deques[neighbor]

                    self.sharing._averaging(averaging_deque)
                else:
                    # Dequeue current messages to skip current iteration.
                    logging.debug("Dummy averaging, emptying queues")
                    for neighbor in self.my_neighbors:
                        _ = self.peer_deques[neighbor].popleft()

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
                    "communication_round": {},
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
            if hasattr(self.sharing, "communication_round"):
                results_dict["communication_round"][
                    iteration + 1
                ] = self.sharing.communication_round

            if rounds_to_train_evaluate == 0 or iteration == 0:
                logging.info("Evaluating on train set.")
                t0 = time.time()
                if iteration != 0:  # We only reset the count after the 1st iteration
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
                self.save_plot(
                    results_dict["test_loss"],
                    "test_loss",
                    "Testing Loss",
                    "Communication Rounds",
                    os.path.join(self.log_dir, "{}_test_loss.png".format(self.rank)),
                )
                t1 = time.time()
                logging.info("Evaluated on train set in %f seconds.", (t1 - t0))

            if self.dataset.__testing__ and (rounds_to_test == 0 or iteration == 0):
                if iteration != 0:  # We only reset the count after the 1st iteration
                    rounds_to_test = self.test_after * change
                logging.info("Evaluating on test set.")
                t0 = time.time()
                ta, tl = self.dataset.test(self.model, self.loss)
                t1 = time.time()
                logging.info("Evaluated on test set in %f seconds.", (t1 - t0))
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
