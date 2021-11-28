
"""Transducer decoder interface module."""

from abc import ABC
from abc import abstractmethod
from typing import Tuple

import torch


class AbsJointNetwork(torch.nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        enc_out: torch.Tensor,
        dec_out: torch.Tensor,
        is_aux: bool = False,
        quantization: bool = False,
    ) -> torch.Tensor:
        """Joint computation of encoder and decoder hidden state sequences.

        Args:
            enc_out: Expanded encoder output state sequences (B, T, 1, D_enc)
            dec_out: Expanded decoder output state sequences (B, 1, U, D_dec)
            is_aux: Whether auxiliary tasks in used.
            quantization: Whether dynamic quantization is used.

        Returns:
            joint_out: Joint output state sequences. (B, T, U, D_out)

        """
        raise NotImplementedError