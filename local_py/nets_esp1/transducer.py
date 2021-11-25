"""Transducer speech recognition model (pytorch)."""

from argparse import ArgumentParser
from argparse import Namespace
from dataclasses import asdict
import logging
import math
import numpy
from typing import Any
from typing import List
from typing import Optional
from typing import Tuple

import chainer
import torch

from espnet.nets.pytorch_backend.initialization import set_forget_bias_to_one
from espnet.nets.asr_interface import ASRInterface
from espnet.nets.beam_search_transducer import BeamSearchTransducer
from espnet.nets.pytorch_backend.nets_utils import get_subsample
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet.nets.pytorch_backend.transducer.utils import get_decoder_input
from espnet.nets.pytorch_backend.transducer.arguments import (
    add_auxiliary_task_arguments,  # noqa: H301
    add_custom_decoder_arguments,  # noqa: H301
    add_custom_encoder_arguments,  # noqa: H301
    add_custom_training_arguments,  # noqa: H301
    add_decoder_general_arguments,  # noqa: H301
    add_encoder_general_arguments,  # noqa: H301
    add_rnn_decoder_arguments,  # noqa: H301
    add_rnn_encoder_arguments,  # noqa: H301
    add_transducer_arguments,  # noqa: H301
)
from espnet.nets.pytorch_backend.transducer.custom_decoder import CustomDecoder
from espnet.nets.pytorch_backend.transducer.custom_encoder import CustomEncoder
from espnet.nets.pytorch_backend.transducer.error_calculator import ErrorCalculator
from espnet.nets.pytorch_backend.transducer.initializer import initializer
from espnet.nets.pytorch_backend.transducer.rnn_decoder import RNNDecoder
from espnet.nets.pytorch_backend.transducer.rnn_encoder import encoder_for
from espnet.nets.pytorch_backend.transducer.transducer_tasks import TransducerTasks
from espnet.nets.pytorch_backend.transducer.utils import get_decoder_input
from espnet.nets.pytorch_backend.transducer.utils import valid_aux_encoder_output_layers
from espnet.nets.pytorch_backend.transformer.attention import (
    MultiHeadedAttention,  # noqa: H301
    RelPositionMultiHeadedAttention,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.mask import target_mask
from espnet.nets.pytorch_backend.transformer.plot import PlotAttentionReport
from espnet.utils.fill_missing_args import fill_missing_args

from espnet.nets.pytorch_backend.transducer.joint_network import JointNetwork

from espnet2.asr.decoder.abs_decoder import AbsDecoder


class RNNT(torch.nn.Module):
    """Transducer modules.

        Args:
            odim: dimension of outputs (vocab__size)
            encoder_output_sizse: number of encoder projection units
            dropout_rate: dropout rate (0.0 ~ 1.0)

    """

    def __init__(
        self,
        odim: int,
        encoder_output_size: int,
        decoder_output_size: int = 256,
        decoder_layers: int = 2,
        decoder_type: str = 'lstm',
        joint_space_size: int = 256,
        joint_activation_type: str ='tanh',
        dropout_rate: float = 0.0,
        dropout_rate_embed: float = 0.0,
        blank_id: int = 0,
        ignore_id: int = -1,
        reduce: bool = True,

    ):
        """Construct a Transducer module."""
        super().__init__()

        self.blank_id = blank_id
        self.ignore_id = ignore_id
        self.dec = RNNDecoder(
            odim,
            decoder_type,
            decoder_layers,
            decoder_output_size,
            decoder_output_size,
            dropout_rate=dropout_rate,
            dropout_rate_embed=dropout_rate_embed,
            blank_id=blank_id,
        )


        self.joint_network = JointNetwork(
            odim, encoder_output_size, decoder_output_size, joint_space_size, joint_activation_type
        )

        from warprnnt_pytorch import RNNTLoss
        self.transducer_loss = RNNTLoss(
            blank=blank_id,
            reduction=('mean' if reduce else 'none'),
        )
        

        self.default_parameters()

    def default_parameters(self):
        """Initialize Transducer model.

        Args:
            model: Transducer model.
            args: Namespace containing model options.

        """
        for name, p in self.named_parameters():
            if any(x in name for x in [ "dec.", "joint_network."]):
                if p.dim() == 1:
                    # bias
                    p.data.zero_()
                elif p.dim() == 2:
                    # linear weight
                    n = p.size(1)
                    stdv = 1.0 / math.sqrt(n)
                    p.data.normal_(0, stdv)
                elif p.dim() in (3, 4):
                    # conv weight
                    n = p.size(1)
                    for k in p.size()[2:]:
                        n *= k
                        stdv = 1.0 / math.sqrt(n)
                        p.data.normal_(0, stdv)


        for i in range(self.dec.dlayers):
            set_forget_bias_to_one(getattr(self.dec.decoder[i], "bias_ih_l0"))
            set_forget_bias_to_one(getattr(self.dec.decoder[i], "bias_hh_l0"))

        



    def compute_transducer_loss(
        self,
        enc_out: torch.Tensor,
        dec_out: torch.Tensor,
        target: torch.Tensor,
        t_len: torch.Tensor,
        u_len: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Transducer loss.

        Args:
            enc_out: Encoder output sequences. (B, T, D_enc)
            dec_out: Decoder output sequences. (B, U, D_dec)
            target: Target label ID sequences. (B, L)
            t_len: Time lengths. (B,)
            u_len: Label lengths. (B,)

        Returns:
            (joint_out, loss_trans):
                Joint output sequences. (B, T, U, D_joint),
                Transducer loss value.

        """
        joint_out = self.joint_network(enc_out.unsqueeze(2), dec_out.unsqueeze(1))

        loss_trans = self.transducer_loss(joint_out, target.to(torch.int32), t_len.to(torch.int32), u_len.to(torch.int32))
        loss_trans /= joint_out.size(0)

        return joint_out, loss_trans

    def loss_fn(self, th_pred, th_target, th_ilen, th_olen) -> torch.Tensor:
        raise NotImplementedError
    
    def forward(self, hs_pad, hlens, ys_pad, ys_lens):
        """Calculate RNNT loss.

        Args:
            hs_pad: batch of padded hidden state sequences (B, Tmax, D)
            hlens: batch of lengths of hidden state sequences (B)
            ys_pad: batch of padded character id sequence tensor (B, Lmax)
            ys_lens: batch of lengths of character sequence (B)
        Returns:
            (joint_out, loss_trans):
                Joint output sequences. (B, T, U, D_joint),
                Transducer loss value.
        """

        dec_in = get_decoder_input(ys_pad, self.blank_id, self.ignore_id)
        self.dec.set_device(hs_pad.device)
        dec_out = self.dec(dec_in)

        joint_out, trans_loss = self.compute_transducer_loss(
            hs_pad, dec_out, ys_pad, hlens, ys_lens
        )
        return trans_loss, joint_out


