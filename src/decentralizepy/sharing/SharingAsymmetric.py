import importlib
import logging
import os

import numpy as np
import torch

from decentralizepy.sharing.Sharing import Sharing


class SharingAsymmetric(Sharing):
    """
    API defining who to share with and what, and what to do on receiving.
    SharingAsymmetric has an additional method "send_all" so that each neighbor can receive a different model.

    """

    def send_all(self, neighbors, averaging_round=0):
        """Sends the model of the current round to all the neighbors

        Args:
            neighbors (int set): The set of neighbors for this round
            averaging_round (int, optional): The averaging round number. Defaults to 0.
        """
        to_send = self.get_data_to_send()
        to_send["CHANNEL"] = "DPSGD"
        to_send["averaging_round"] = averaging_round

        for i, neighbor in enumerate(neighbors):
            self.communication.send(neighbor, to_send)
            if i == 0:
                # We save one of the sent model
                self.check_and_save_sent_model(to_send["params"], neighbor)

    def check_and_save_sent_model(self, model_weights, target):
        """Makes necessary checks to save sent model for future attacks.

        Args:
            model_weights (_type_): _description_   #TODO
            target (_type_): _description_          #TODO
        """
        if self.save_models_for_attacks > 0:
            assert (
                self.training_iteration is not None
            ), "SharingAssymetric's training iteration is None, when it should be an int."
            # When self.save_all_models is False, only the first `self.nb_models_to_save` models are saved on each machine.
            # This means only threshold/classifier attack can be performed, and models will be missing.
            # It may break attacks (in particular linkability).
            if (not self.save_all_models) and (self.uid >= self.nb_models_to_save):
                return
            # Ensure we are at the right iteration
            if (
                self.training_iteration == 0
                or (self.training_iteration + 1) % self.save_models_for_attacks == 0
            ):
                model_name = (
                    f"model_it{self.training_iteration+1}_{self.uid}_to{target}.npy"
                )
                model_path = os.path.join(self.model_save_folder, model_name)
                logging.debug("Saving model parameters to %s", model_path)
                with open(model_path, "wb") as f:
                    np.save(f, model_weights)

    def _averaging(self, peer_deques):
        """
        Averages the received model with the local model

        """
        with torch.no_grad():
            total = dict()
            weight_total = 0
            for i, n in enumerate(peer_deques):
                data = peer_deques[n].popleft()
                degree, iteration, averaging_round = (
                    data["degree"],
                    data["iteration"],
                    data["averaging_round"],
                )
                del data["averaging_round"]
                del data["degree"]
                del data["iteration"]
                del data["CHANNEL"]
                logging.debug(
                    f"Averaging model from neighbor {n} of iteration {iteration}, averaging round {averaging_round}"
                )
                data = self.deserialized_model(data)
                # Metro-Hastings
                # TODO: Generalize this to arbitrary communication matrix?
                # In this case it should be handled by the PeerSampler.
                weight = 1 / (max(len(peer_deques), degree) + 1)
                weight_total += weight
                for key, value in data.items():
                    if key in total:
                        total[key] += value * weight
                    else:
                        total[key] = value * weight

            for key, value in self.model.state_dict().items():
                total[key] += (1 - weight_total) * value  # Metro-Hastings

        self.model.load_state_dict(total)
        self._post_step()
        self.communication_round += 1

    def get_data_to_send(self, degree=None):
        data_to_send = super().get_data_to_send(degree)
        data_to_send["iteration"] = (
            self.training_iteration
        )  # When we have multiple averaging rounds.
        return data_to_send

    def __init__(
        self,
        rank,
        machine_id,
        communication,
        mapping,
        graph,
        model,
        dataset,
        log_dir,
        compress=False,
        compression_package=None,
        compression_class=None,
        float_precision=None,
        save_models_for_attacks=-1,
        save_all_models=True,
        nb_models_to_save=1,
    ):
        """
        Constructor

        Parameters
        ----------
        rank : int
            Local rank
        machine_id : int
            Global machine id
        communication : decentralizepy.communication.Communication
            Communication module used to send and receive messages
        mapping : decentralizepy.mappings.Mapping
            Mapping (rank, machine_id) -> uid
        graph : decentralizepy.graphs.Graph
            Graph representing neighbors
        model : decentralizepy.models.Model
            Model to train
        dataset : decentralizepy.datasets.Dataset
            Dataset for sharing data. Not implemented yet! TODO
        log_dir : str
            Location to write shared_params (only writing for 2 procs per machine)
        compress: bool, default False
            Wether to apply a compression method or not.
        compression_package : str
            Import path of a module that implements the compression.Compression.Compression class
        compression_class : str
            Name of the compression class inside the compression package
        float_precision:

        save_models_for_attacks: int, default -1
            The interval at which a sent model must be logged.
        save_all_models: bool, default True
            Whether to save all models for the attack, or only models for the first of each node.
        nb_models_to_save: int, default 1
            How many nodes' outgoing models are to be saved. Requires `save_all_models` to be False.
        """
        super().__init__(
            rank=rank,
            machine_id=machine_id,
            communication=communication,
            mapping=mapping,
            graph=graph,
            model=model,
            dataset=dataset,
            log_dir=log_dir,
            compress=compress,
            compression_package=compression_package,
            compression_class=compression_class,
            float_precision=float_precision,
        )

        self.training_iteration = None
        self.save_models_for_attacks = save_models_for_attacks
        self.save_all_models = save_all_models
        self.nb_models_to_save = nb_models_to_save
        logging.info(
            "Logging of all models set to %s - nb_models_to_save set to %s",
            self.save_all_models,
            self.nb_models_to_save,
        )
        self.model_save_folder = os.path.join(
            log_dir, f"attacked_model/machine{machine_id}/{rank}/"
        )

        # Only create the folder if the model will be logged.
        if self.save_all_models or self.uid < self.nb_models_to_save:
            if not os.path.exists(self.model_save_folder):
                os.makedirs(self.model_save_folder)
            else:
                logging.warning(
                    "The directory %s already exists", self.model_save_folder
                )
