
"""Transducer decoder interface module."""

from abc import ABC
from abc import abstractmethod
from typing import Tuple

import torch

from espnet2.asr.beam_search.scorer_interface import ScorerInterface


class AbsDecoder(torch.nn.Module, ScorerInterface, ABC):
    @abstractmethod
    def forward(
        self,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


