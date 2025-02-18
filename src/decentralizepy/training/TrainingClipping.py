import logging

import torch
import torch.nn.utils

from decentralizepy.training.Training import Training


class TrainingClipping(Training):

    def __init__(
        self,
        rank,
        machine_id,
        mapping,
        model,
        optimizer,
        loss,
        log_dir,
        rounds="",
        full_epochs="",
        batch_size="",
        shuffle="",
        clipping_norm=None,
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
        model : torch.nn.Module
            Neural Network for training
        optimizer : torch.optim
            Optimizer to learn parameters
        loss : function
            Loss function
        log_dir : str
            Directory to log the model change.
        rounds : int, optional
            Number of steps/epochs per training call
        full_epochs : bool, optional
            True if 1 round = 1 epoch. False if 1 round = 1 minibatch
        batch_size : int, optional
            Number of items to learn over, in one batch
        shuffle : bool
            True if the dataset should be shuffled before training.
        clipping_norm: optional float.
            The clipping norm to use during training. No clipping if None.
        """
        super(TrainingClipping, self).__init__(
            rank=rank,
            machine_id=machine_id,
            mapping=mapping,
            model=model,
            optimizer=optimizer,
            loss=loss,
            log_dir=log_dir,
            rounds=rounds,
            full_epochs=full_epochs,
            batch_size=batch_size,
            shuffle=shuffle,
        )
        self.clipping_norm = clipping_norm

    def trainstep(self, data, target):
        """
        One training step on a minibatch.

        Parameters
        ----------
        data : any
            Data item
        target : any
            Label

        Returns
        -------
        int
            Loss Value for the step

        """
        self.model.zero_grad()
        output = self.model(data)
        loss_val = self.loss(output, target)
        loss_val.backward()
        if self.clipping_norm is not None:
            logging.debug("Clipping with norm %s", self.clipping_norm)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipping_norm)
        self.optimizer.step()
        return loss_val.item()
