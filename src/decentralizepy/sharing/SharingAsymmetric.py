import importlib
import logging

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

        for neighbor in neighbors:
            self.communication.send(neighbor, to_send)

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
        # self.communication_round +=1