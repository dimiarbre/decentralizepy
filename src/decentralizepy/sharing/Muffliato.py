import copy
import logging

import numpy as np

from decentralizepy.sharing.SharingAsymmetric import SharingAsymmetric


class Muffliato(SharingAsymmetric):
    """
    API defining who to share with and what, and what to do on receiving
    This is an implementation of https://proceedings.neurips.cc/paper_files/paper/2022/hash/65d32185f73cbf4535449a792c63926f-Abstract-Conference.html
    """

    def send_all(self, neighbors, averaging_round=0):
        """Sends the data to all the neighbors

        Args:
            neighbors (int set): The neighbors to send the current model to
            averaging_round (int, optional): The step number, needed for Muffliato when averaging more than once. Defaults to 0.
        """
        logging.info(
            f"Sending to all the neighbors {neighbors} for averaging step {averaging_round}"
        )
        current_model_data = self.get_data_to_send()
        current_model_data["CHANNEL"] = "DPSGD"
        current_model_data["averaging_round"] = averaging_round
        to_send = copy.deepcopy(current_model_data)

        if averaging_round == 0:
            noise = np.random.normal(
                0, self.noise_std, current_model_data["params"].shape
            )  # Generate the model noise

            self.generated_noise_std = np.std(noise)  # Stats of the noise
            logging.info(
                f"Total noise std: {self.generated_noise_std}, expected std {self.noise_std}."
            )
            logging.debug(f"Noise shape : {noise.shape}.")

            to_send["params"] += noise  # Noise the sent parameters
            # Then load this noised model as the local model.
            self.model.load_state_dict(self.deserialized_model(to_send))
        # TODO: This implementation will not work with compression algorithms, since we noise the compressed value here
        # (the result of `get_data_to_send()`)

        for i, neighbor in enumerate(neighbors):
            self.communication.send(neighbor, to_send)

            if (
                averaging_round == 0 and i == 0
            ):  # We attack the model sent to one neighbor, as if an attacker saw the message.
                # TODO: probably randomize which neighbor is attacked to avoid attacking always the same one?
                # logging.debug(f"Attacking model sent to {neighbor}.")
                # TODO: Make the attack at the correct time
                pass

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
        noise_std=0,
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
        noise_std: int, default 0
            The std of the Gaussian noise generated by Muffliato
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
        )
        self.noise_std = noise_std
        self.generated_noise_std = None