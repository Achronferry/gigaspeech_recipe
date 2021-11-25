from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F



# From: https://github.com/hasangchun/Transformer-Transducer/tree/main/transformer_transducer

def build_transducer(
        num_vocabs: int,
        model_type: str  = "T-T", 
        input_size: int = 80,
        model_dim: int = 512,
        ff_dim: int = 2048,
        num_audio_layers: int = 18,
        num_label_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.3,
        max_len: int = 5000,
        pad_id: int = 0,
        sos_id: int = 1,
        eos_id: int = 2,
) -> Transducer:

    assert model_type in ["RNN-T", "T-T"]
    
    audio_encoder = AudioEncoder(input_size, model_dim, ff_dim,
                                num_audio_layers, num_heads, dropout, max_len)

    decoder = LabelEncoder( num_vocabs, model_dim, ff_dim,
                                num_label_layers, num_heads, dropout, max_len, pad_id, sos_id, eos_id)
    return TransformerTransducer(encoder, decoder, num_vocabs, model_dim << 1, model_dim)




class Transducer(nn.Module):
    """
    Transformer-Transducer is that every layer is identical for both audio and label encoders.
    Unlike the basic transformer structure, the audio encoder and label encoder are separate.
    So, the alignment is handled by a separate forward-backward process within the RNN-T architecture.
    And we replace the LSTM encoders in RNN-T architecture with Transformer encoders.
    Args:
        audio_encoder (AudioEncoder): Instance of audio encoder
        label_encoder (LabelEncoder): Instance of label encoder
        num_vocabs (int): the number of vocabulary
        output_size (int): the number of features combined output of audio and label encoders (default : 1024)
        inner_size (int): the number of inner features (default : 512)
    Inputs: inputs, input_lens, targets, targets_lens
        - **inputs** (torch.FloatTensor): A input sequence passed to audio encoder. Typically inputs will be a padded
            `FloatTensor` of size ``(batch, dimension, seq_length)``.
        - **input_lens** (torch.LongTensor): The length of input tensor. ``(batch)``
        - **targets** (torch.LongTensor): A target sequence passed to label encoder. Typically inputs will be a padded
            `LongTensor` of size ``(batch, target_length)``
        - **targets_lens** (torch.LongTensor): The length of target tensor. ``(batch)``
    Returns: output
        - **output** (torch.FloatTensor): Result of model predictions.
    """
    def __init__(
            self,
            audio_encoder: AudioEncoder,
            label_encoder: LabelEncoder,
            num_vocabs: int,
            output_size: int = 1024,
            inner_size: int = 512,
    ) -> None:
        super(Transducer, self).__init__()
        self.audio_encoder = audio_encoder
        self.label_encoder = label_encoder
        self.joint = JointNet(num_vocabs, output_size, inner_size)

    def forward(
            self,
            inputs: Tensor,
            input_lens: Tensor,
            targets: Tensor,
            targets_lens: Tensor,
    ) -> Tensor:
        """
        Forward propagate a `inputs, targets` for transformer transducer.
        Args:
            inputs (torch.FloatTensor): A input sequence passed to audio encoder. Typically inputs will be a padded
                `FloatTensor` of size ``(batch, dimension, seq_length)``.
            input_lens (torch.LongTensor): The length of input tensor. ``(batch)``
            targets (torch.LongTensor): A target sequence passed to label encoder. Typically inputs will be a padded
                `LongTensor` of size ``(batch, target_length)``
            targets_lens (torch.LongTensor): The length of target tensor. ``(batch)``
        Returns:
            **output** (Tensor): ``(batch, seq_length, num_vocabs)``
        """
        audio_output = self.audio_encoder(inputs, input_lens)
        label_output = self.label_encoder(targets, targets_lens)

        output = self.joint(audio_output, label_output)

        return output

    @torch.no_grad()
    def decode(self, audio_outputs: Tensor, max_lens: int) -> Tensor:
        batch = audio_outputs.size(0)
        y_hats = list()

        targets = torch.LongTensor([self.label_encoder.sos_id] * batch)
        if torch.cuda.is_available():
            targets = targets.cuda()

        for i in range(max_lens):
            label_output = self.label_encoder(targets, None)
            label_output = label_output.squeeze(1)
            audio_output = audio_outputs[:, i, :]
            output = self.joint(audio_output, label_output)
            targets = output.max(1)[1]
            y_hats.append(targets)

        y_hats = torch.stack(y_hats, dim=1)

        return y_hats  # (B, T)

    @torch.no_grad()
    def recognize(self, inputs: Tensor, inputs_lens: Tensor) -> Tensor:
        audio_outputs = self.audio_encoder(inputs, inputs_lens)
        max_lens = audio_outputs.size(1)

        return self.decode(audio_outputs, max_lens)


class AudioEncoder(nn.Module):
    """
    Converts the audio signal to higher feature values
    Args:
        input_size (int): dimension of input vector (default : 80)
        model_dim (int): the number of features in the audio encoder (default : 512)
        ff_dim (int): the number of features in the feed forward layers (default : 2048)
        num_layers (int): the number of audio encoder layers (default: 18)
        num_heads (int): the number of heads in the multi-head attention (default: 8)
        dropout (float): dropout probability of audio encoder (default: 0.1)
        max_len (int): Maximum length to use for positional encoding (default : 5000)
    Inputs: inputs, inputs_lens
        - **inputs**: Parsed audio of batch size number
        - **inputs_lens**: Tensor of sequence lengths
    Returns: outputs
        - **outputs**: Tensor containing higher feature values
    """
    def __init__(
            self,
            input_size: int = 80,
            model_dim: int = 512,
            ff_dim: int = 2048,
            num_layers: int = 18,
            num_heads: int = 8,
            dropout: float = 0.1,
            max_len: int = 5000,
    ) -> None:
        super(AudioEncoder, self).__init__()
        self.input_size = input_size
        self.model_dim = model_dim
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.input_dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(model_dim)
        self.positional_encoding = PositionalEncoding(model_dim, max_len)
        self.input_fc = nn.Linear(input_size, model_dim)
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(model_dim, ff_dim, num_heads, dropout) for _ in range(num_layers)]
        )

    def forward(
            self,
            inputs: Tensor,
            inputs_lens: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward propagate a `inputs` for audio encoder.
        Args:
            inputs (torch.FloatTensor): A input sequence passed to audio encoder. Typically inputs will be a padded
                `FloatTensor` of size ``(batch, dimension, seq_length)``.
            inputs_lens (torch.LongTensor): The length of input tensor. ``(batch)``
        Returns:
            **outputs** (Tensor): ``(batch, seq_length, dimension)``
        """
        inputs = inputs.transpose(1, 2)
        seq_len = inputs.size(1)

        self_attn_mask = get_attn_pad_mask(inputs, inputs_lens, seq_len)

        inputs = self.input_fc(inputs) + self.positional_encoding(seq_len)
        outputs = self.input_dropout(inputs)

        for encoder_layer in self.encoder_layers:
            outputs, _ = encoder_layer(outputs, self_attn_mask)

        return outputs



class LabelEncoder(nn.Module):
    """
    Converts the label to higher feature values
    Args:
        device (torch.device): flag indication whether cpu or cuda
        num_vocabs (int): the number of vocabulary
        model_dim (int): the number of features in the label encoder (default : 512)
        ff_dim (int): the number of features in the feed forward layers (default : 2048)
        num_layers (int): the number of label encoder layers (default: 2)
        num_heads (int): the number of heads in the multi-head attention (default: 8)
        dropout (float): dropout probability of label encoder (default: 0.1)
        max_len (int): Maximum length to use for positional encoding (default : 5000)
        pad_id (int): index of padding (default: 0)
        sos_id (int): index of the start of sentence (default: 1)
        eos_id (int): index of the end of sentence (default: 2)
    Inputs: inputs, inputs_lens
        - **inputs**: Ground truth of batch size number
        - **inputs_lens**: Tensor of target lengths
    Returns: outputs
        - **outputs**: Tensor containing higher feature values
    """
    def __init__(
            self,
            num_vocabs: int,
            model_dim: int = 512,
            ff_dim: int = 2048,
            num_layers: int = 2,
            num_heads: int = 8,
            dropout: float = 0.1,
            max_len: int = 5000,
            pad_id: int = 0,
            sos_id: int = 1,
            eos_id: int = 2,
    ) -> None:
        super(LabelEncoder, self).__init__()
        self.embedding = nn.Embedding(num_vocabs, model_dim)
        self.scale = np.sqrt(model_dim)
        self.positional_encoding = PositionalEncoding(model_dim, max_len)
        self.input_dropout = nn.Dropout(p=dropout)
        self.pad_id = pad_id
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(model_dim, ff_dim, num_heads, dropout) for _ in range(num_layers)]
        )

    def forward(
            self,
            inputs: Tensor,
            inputs_lens: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward propagate a `inputs` for label encoder.
        Args:
            inputs (torch.LongTensor): A input sequence passed to label encoder. Typically inputs will be a padded
                `LongTensor` of size ``(batch, target_length)``
            inputs_lens (torch.LongTensor): The length of input tensor. ``(batch)``
        Returns:
            **outputs** (Tensor): ``(batch, seq_length, dimension)``
        """
        self_attn_mask = None
        batch = inputs.size(0)

        if len(inputs.size()) == 1:  # validate, evaluation
            inputs = inputs.unsqueeze(1)
            target_lens = inputs.size(1)

            embedding_output = self.embedding(inputs) * self.scale
            positional_encoding_output = self.positional_encoding(target_lens)
            inputs = embedding_output + positional_encoding_output

        else:  # train
            inputs = inputs[inputs != self.eos_id].view(batch, -1)
            target_lens = inputs.size(1)

            embedding_output = self.embedding(inputs) * self.scale
            positional_encoding_output = self.positional_encoding(target_lens)
            inputs = embedding_output + positional_encoding_output

            self_attn_mask = get_attn_pad_mask(inputs, inputs_lens, target_lens)

        outputs = self.input_dropout(inputs)

        for encoder_layer in self.encoder_layers:
            outputs, _ = encoder_layer(outputs, self_attn_mask)

        return outputs


class JointNet(nn.Module):
    """
    Combine the audio encoder and label encoders.
    Convert them into log probability values for each word.
    Args:
        num_vocabs (int): the number of vocabulary
        output_size (int): the number of features combined output of audio and label encoders (default : 1024)
        inner_size (int): the number of inner features (default : 512)
    Inputs: audio_encoder, label_encoder
        - **audio_encoder**: Audio encoder output
        - **label_encoder**: Label encoder output
    Returns: output
        - **output**: Tensor expressing the log probability values of each word
    """
    def __init__(
            self,
            num_vocabs: int,
            output_size: int = 1024,
            inner_size: int = 512,
    ) -> None:
        super(JointNet, self).__init__()
        self.fc1 = nn.Linear(output_size, inner_size)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(inner_size, num_vocabs)

    def forward(
            self,
            audio_encoder: Tensor,
            label_encoder: Tensor,
    ) -> Tensor:
        if audio_encoder.dim() == 3 and label_encoder.dim() == 3:  # Train
            seq_lens = audio_encoder.size(1)
            target_lens = label_encoder.size(1)

            audio_encoder = audio_encoder.unsqueeze(2)
            label_encoder = label_encoder.unsqueeze(1)

            audio_encoder = audio_encoder.repeat(1, 1, target_lens, 1)
            label_encoder = label_encoder.repeat(1, seq_lens, 1, 1)

        output = torch.cat((audio_encoder, label_encoder), dim=-1)

        output = self.fc1(output)
        output = self.tanh(output)
        output = self.fc2(output)

        output = F.log_softmax(output, dim=-1)

        return output

if __name__=='__main__':
    pass