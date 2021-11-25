import torch
import torch.nn as nn
from dataclasses import asdict

from espnet.nets.pytorch_backend.transducer.loss import TransLoss
from espnet.nets.pytorch_backend.transducer.rnn_decoder import DecoderRNNT
from espnet.nets.pytorch_backend.transducer.utils import prepare_loss_inputs
from espnet.nets.beam_search_transducer import BeamSearchTransducer


class Transducer(nn.Module):
    def __init__(self, in_size, dunits, dtype, num_vocabs, dlayers, 
                embed_dim, joint_dim, blank_id=0, ignore_id=-1, dropout=0.0):
        super().__init__()

        self.predictor = DecoderRNNT(num_vocabs, dtype, dlayers, dunits, 
                                    blank_id, embed_dim, dropout, dropout_embed=dropout)
        self.joint = JointNet(num_vocabs, in_size, dunits, joint_dim)
        self.loss = TransLoss("warp-transducer", blank_id)
        self.blank_id = blank_id
        self.ignore_id = ignore_id


    
    def forward(self, hs_pad, hs_len, ys_pad):
        # 1.5. transducer preparation related
        ys_in_pad, ys_out_pad, target, pred_len, target_len = prepare_loss_inputs(
            ys_pad, hs_len, self.blank_id, self.ignore_id
        )

        pred_pad = self.predictor(hs_pad, ys_in_pad)

        z = self.joint(hs_pad.unsqueeze(2), pred_pad.unsqueeze(1))
        loss_trans = self.loss(z, target, pred_len, target_len)

        # pad for nn.Parallel
        return loss_trans, z
    
    
    def recognize(self, h, **beam_kargs):
        """Recognize input features.

        Args:
            x (ndarray): input acoustic feature (T, D)
            beam_search (class): beam search class

        Returns:
            nbest_hyps (list): n-best decoding results

        """
        beam_search_transducer = BeamSearchTransducer(
            decoder=self.predictor,
            joint_network=self.joint,
            **beam_kargs
        )

        nbest_hyps = beam_search_transducer(h)
        return [asdict(n) for n in nbest_hyps]
        




class JointNet(nn.Module):
    """Transducer joint network module.

    Args:
        joint_space_size: Dimension of joint space
        joint_activation_type: Activation type for joint network

    """

    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        decoder_output_size: int,
        joint_space_size: int,
        joint_activation_type: str = 'Tanh',
    ):
        """Joint network initializer."""
        super().__init__()

        self.lin_enc = torch.nn.Linear(encoder_output_size, joint_space_size)
        self.lin_dec = torch.nn.Linear(
            decoder_output_size, joint_space_size, bias=False
        )

        self.lin_out = torch.nn.Linear(joint_space_size, vocab_size)

        assert joint_activation_type in ['Tanh', 'Hardtanh', 'ReLU']
        self.joint_activation = getattr(nn, joint_activation_type)()

    def forward(
        self, h_enc: torch.Tensor, h_dec: torch.Tensor, is_aux: bool = False
    ) -> torch.Tensor:
        """Joint computation of z.

        Args:
            h_enc: Batch of expanded hidden state (B, T, 1, D_enc)
            h_dec: Batch of expanded hidden state (B, 1, U, D_dec)

        Returns:
            z: Output (B, T, U, vocab_size)

        """
        if is_aux:
            z = self.joint_activation(h_enc + self.lin_dec(h_dec))
        else:
            z = self.joint_activation(self.lin_enc(h_enc) + self.lin_dec(h_dec))
        z = self.lin_out(z)

        return z

if __name__ == '__main__':
    mdl = Transducer(256, 320,'lstm',10, 2, 128, 512)
    ys_pad = torch.nn.utils.rnn.pad_sequence(
        [torch.randint(0, 10, (l,)) for l in [5,4,6]], batch_first=True, padding_value=-1)
    feats = torch.rand((3, 50, 256), dtype=torch.float)
    ilen = torch.tensor([45, 50 ,31]) 

    loss, outputs = mdl(feats, ilen, ys_pad) # 1, (3, 50, 7, 10)

    mdl.eval()
    with torch.no_grad():
        pred = mdl.recognize(torch.rand((50, 256), dtype=torch.float), beam_size=10, nbest=3)

    print(pred[0].keys())
    print([i['score'] for i in pred])
    print([i['yseq'] for i in pred])
    print([len(i['dec_state']) for i in pred])
    print([i['dec_state'][0].shape for i in pred])
    print([i['lm_state'] for i in pred])
