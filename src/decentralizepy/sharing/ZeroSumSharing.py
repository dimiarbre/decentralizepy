import importlib
import logging
import copy
import numpy as np

import torch

from decentralizepy.sharing.SharingAsymmetric import SharingAsymmetric

class ZeroSumSharing(SharingAsymmetric):
    """
    API defining who to share with and what, and what to do on receiving

    """

    def send_all(self,neighbors,averaging_round= 0):
        logging.info(f"Sending to all the neighbors {neighbors} for averaging step {averaging_round}")
        current_model_data = self.get_data_to_send()
        current_model_data["CHANNEL"] = "DPSGD"
        current_model_data["averaging_round"] = averaging_round

        if averaging_round==0: # Only noise on the first averaging round
            nb_neighbors = len(neighbors)
            std_to_gen = np.sqrt(nb_neighbors/(nb_neighbors-1)) * self.noise_std
            #TODO: Add self noise
            noises = np.random.normal(0,std_to_gen,(len(neighbors),) + current_model_data["params"].shape)

            avg_noise = np.average(noises,axis = 0)
            #Normalize the noise
            noises -=  avg_noise

            self.generated_noise_std = np.std(noises)
            logging.info(f"Total noise std: {self.generated_noise_std}, expected std {self.noise_std}. Generated with std {std_to_gen}.")
            logging.debug(f"Noise shape : {noises.shape}. Avg shape : {avg_noise.shape}.")


        for i,neighbor in enumerate(neighbors):
            to_send = copy.deepcopy(current_model_data)
            if averaging_round==0:
                noise = noises[i]
                logging.debug(f"Current noise shape : {noise.shape}. Current noise avg : {np.average(noise)}.")
                to_send["params"] += noise
            self.communication.send(neighbor, to_send)

            # if i == 0 and averaging_round == 0:
            #     # We attack the model sent to one neighbor, since it is what the attacker will be able to see
            #     logging.debug(f"Attacking model sent to {neighbor}.")

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
        compression_class= None,
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

        """
        self.noise_std = noise_std
        self.generated_noise_std = None
        super().__init__(rank,
                        machine_id,
                        communication,
                        mapping,
                        graph,
                        model,
                        dataset,
                        log_dir,
                        compress = compress,
                        compression_package = compression_package,
                        compression_class = compression_class,
                        )
