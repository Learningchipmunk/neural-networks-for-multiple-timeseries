import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchscale.architecture.config import RetNetConfig
from torchscale.architecture.retnet import RetNetDecoder

class TabRetNet(nn.Module):
  def __init__(
        self,
        config_path,
        input_dim,
        output_dim,
        embed_dropout_rate=0.1,
        is_multiple_segments_input=False,
        is_bidir=False,
        max_num_era_in_seq=4,
        **kwargs
    ):
    super().__init__(**kwargs)

    # Defining ReNet Decoder
    with open(config_path) as f:
        retnet_args = json.load(f)
    config = RetNetConfig(**retnet_args)
    self.retnet = RetNetDecoder(config)
    self.is_multiple_segments_input = is_multiple_segments_input
    self.is_bidir = is_bidir

    # Scale the embeddings to hidden dim
    self.input_embed = torch.nn.Linear(
            input_dim,
            config.decoder_embed_dim,
            bias=True,
        )

    # # Add segment embeddings for eras
    # self.scaled_embed_layer_norm  = nn.LayerNorm(config.decoder_embed_dim)
    # self.segment_embed_layer_norm = nn.LayerNorm(config.decoder_embed_dim)
    if is_multiple_segments_input:
        self.segment_embeddings = torch.nn.Embedding(max_num_era_in_seq+1, config.decoder_embed_dim, padding_idx=0)# +1 for padding value

    # Add dropout before regressor
    self.dropout_embed = nn.Dropout(p=embed_dropout_rate)

    # Defining Regressor for Prediction
    decoder_embed_dim = int(2*config.decoder_embed_dim) if self.is_bidir else config.decoder_embed_dim
    self.regressor = torch.nn.Linear(
            decoder_embed_dim,
            output_dim,
            bias=True,
        )

    # Sigmoid because target is gaussian and its values is between [0,1]
    self.act = nn.Sigmoid()

  def forward(self, input_embeddings, segment_masks=None):
    '''Takes input embeddings and returns the prediction. It is the main forward pass of the model.

    Args:
        input_embeddings (_type_): The input embeddings (batch_size, seq_len, input_dim).
        segment_masks (_type_, optional): Mask that indicates each embedding to which segment it belongs in the sequence (batch_size, seq_len, 1). Defaults to None.
    '''
    scaled_input_embeds = self.input_embed(input_embeddings)

    # Encodes eras as different segments of time
    if hasattr(self, 'segment_embeddings') and segment_masks is not None and self.is_multiple_segments_input:
        # scaled_input_embeds = self.scaled_embed_layer_norm(scaled_input_embeds) + self.segment_embed_layer_norm(self.segment_embeddings(segment_label))
        scaled_input_embeds = scaled_input_embeds + self.segment_embeddings(segment_masks.squeeze(-1))#.squeeze(-2)

    # Dummy variable, because we don't have a previous output => we use directly token embeddings
    tokens = torch.ones_like(scaled_input_embeds[:,:,0])

    # Getting new embeddings from ReNet
    embeds = self.retnet(
        prev_output_tokens=tokens,
        token_embeddings=scaled_input_embeds,
        features_only=True,
      )[0]

    if self.is_bidir:
        # Flips the sequence
        scaled_input_embeds_flipped = torch.flip(scaled_input_embeds, dims=[1])

        # Getting new embeddings from ReNet
        embeds_2 = self.retnet(
            prev_output_tokens=tokens,
            token_embeddings=scaled_input_embeds_flipped,
            features_only=True,
        )[0]
        embeds = torch.cat([embeds, embeds_2], dim=-1)

    # Adding dropout before regressor
    if hasattr(self, 'dropout_embed'):#Keeps compatibility with older models (they don't have `dropout_embed`)
        embeds = self.dropout_embed(embeds)

    return 1.1*self.act(self.regressor(embeds)).squeeze(-1)-0.05