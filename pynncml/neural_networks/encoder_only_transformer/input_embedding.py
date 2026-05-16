import torch
import torch.nn as nn
from pynncml import neural_networks
from pynncml.neural_networks.normalization import InputNormalization

class InputEmbedding(nn.Module):
    def __init__(self, normalization_cfg: neural_networks.InputNormalizationConfig,
                 dynamic_input_size,
                 metadata_input_size,
                 d_model,
                 protocol_n_features,
                 metadata_n_features,
                 num_protocols
                 ):
        super().__init__()
        self.normalization = InputNormalization(normalization_cfg)
        self.protocol_n_features = protocol_n_features
        self.metadata_n_features = metadata_n_features
        self.dynamic_n_features = d_model - protocol_n_features - metadata_n_features
        assert self.dynamic_n_features > 0, "d_model too small for protocol+metadata features"

        # dynamic projections
        self.dynamic_linear = nn.Linear(dynamic_input_size, self.dynamic_n_features)  
        
        # metadata projection
        self.metadata_linear = nn.Linear(metadata_input_size, self.metadata_n_features)

        # protocol embedding
        self.protocol_emb = nn.Embedding(num_protocols, self.protocol_n_features)

    def forward(self, dynamic_data, metadata, protocol_id):
        """
        dynamic_data: [B, T, 180]
        metadata:     [B, 2]
        protocol_id:  [B]
        Returns:      [B, T, d_model]
        """
        B, T, _ = dynamic_data.shape

        input_tensor, input_meta_tensor = self.normalization(dynamic_data, metadata)
        
        # ---- dynamic data projection ----
        dyn_emb = self.dynamic_linear(input_tensor)   # [B, T, dynamic_n_features]

        # ---- metadata projection ----
        meta_emb = self.metadata_linear(input_meta_tensor)  # [B, metadata_n_features]
        meta_emb = meta_emb.unsqueeze(1).expand(-1, T, -1)  # [B, T, metadata_n_features]

        # ---- protocol embedding ----
        proto_emb = self.protocol_emb(protocol_id.long())  # [B, protocol_n_features]
        proto_emb = proto_emb.unsqueeze(1).expand(-1, T, -1)  # [B, T, protocol_n_features]

        # Concatenate along feature dimension
        x = torch.cat([proto_emb, dyn_emb, meta_emb], dim=-1)  # [B, T, d_model]
        return x
    



# this for: class InputEmbedding(nn.Module):
#self.dynamic_linear_inst = nn.Linear(dynamic_input_size, self.dynamic_n_features)  # 180 -> dyn
##self.dynamic_linear_avg = nn.Linear(1, self.dynamic_n_features)                   # avg attenuation
##self.dynamic_linear_mm = nn.Linear(2, self.dynamic_n_features)                    # min-max attenuation
#self.dynamic_linear_avg = nn.Linear(2, self.dynamic_n_features)                    # [avg_rsl, avg_tsl]
#self.dynamic_linear_mm = nn.Linear(4, self.dynamic_n_features)                     # [max_rsl, min_rsl, max_tsl, min_tsl] 

# this for forward:
'''
        #inst_mask = protocol_id <= 12
        #avg_mask = protocol_id == 12
        #mm_mask = protocol_id == 13

        #if inst_mask.any():
        #    dyn_emb[inst_mask] = self.dynamic_linear_inst(input_tensor[inst_mask])

#        if avg_mask.any():
#            avg_input = input_tensor[avg_mask, :, 0:1]   # [N_avg, T, 1]
#            dyn_emb[avg_mask] = self.dynamic_linear_avg(avg_input)

#        if mm_mask.any():
#            mm_input = torch.stack(
#                [
#                    input_tensor[mm_mask, :, 0],    # first value
#                    input_tensor[mm_mask, :, 60]    # value at index 60
#                ],
#                dim=-1
#            )  # [N_mm, T, 2]
#            dyn_emb[mm_mask] = self.dynamic_linear_mm(mm_input)
        if avg_mask.any():
            avg_input = torch.stack(
                [
                    input_tensor[avg_mask, :, 0],    # avg_rsl
                    input_tensor[avg_mask, :, 90],   # avg_tsl
                ],
                dim=-1
            )  # [N_avg, T, 2]
            dyn_emb[avg_mask] = self.dynamic_linear_avg(avg_input)

        if mm_mask.any():
            mm_input = torch.stack(
                [
                    input_tensor[mm_mask, :, 0],     # max_rsl
                    input_tensor[mm_mask, :, 45],    # min_rsl
                    input_tensor[mm_mask, :, 90],    # max_tsl
                    input_tensor[mm_mask, :, 135],   # min_tsl
                ],
                dim=-1
            )  # [N_mm, T, 4]
            dyn_emb[mm_mask] = self.dynamic_linear_mm(mm_input)'''

'''
class DynamicEncoder(nn.Module):
    def __init__(
        self,
        dynamic_input_size: int,
        d_dynamic_n_features: int,
        protocol_n_features: int,
        num_protocols: int
    ):
        super().__init__()

        self.d_large = 3 * d_dynamic_n_features

        # FC(180 -> d_large)
        self.fc1 = nn.Linear(dynamic_input_size, self.d_large)

        # Separate protocol embedding for AdaIN style generation
        self.protocol_style_emb = nn.Embedding(num_protocols, protocol_n_features)

        # Produce AdaIN parameters (scale and bias)
        self.style_affine = nn.Linear(protocol_n_features, 2 * self.d_large)

        # Non-linearity
        self.prelu = nn.PReLU(num_parameters=self.d_large)

        # FC(d_large -> d_dynamic_n_features)
        self.fc2 = nn.Linear(self.d_large, d_dynamic_n_features)

    def adain(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """
        x:     [B, T, d_large]
        style: [B, 2*d_large]
        """
        y_s, y_b = torch.chunk(style, 2, dim=-1)   # [B, d_large], [B, d_large]

        y_s = y_s.unsqueeze(1)   # [B, 1, d_large]
        y_b = y_b.unsqueeze(1)   # [B, 1, d_large]

        mu = x.mean(dim=-1, keepdim=True)                              # [B, T, 1]
        sigma = x.std(dim=-1, keepdim=True, unbiased=False) + 1e-6    # [B, T, 1]

        x_norm = (x - mu) / sigma

        return y_s * x_norm + y_b

    def forward(self, dynamic_data: torch.Tensor, protocol_id: torch.Tensor) -> torch.Tensor:
        """
        dynamic_data: [B, T, 180]
        protocol_id:  [B]

        returns:      [B, T, d_dynamic_n_features]
        """
        x = self.fc1(dynamic_data)   # [B, T, d_large]

        protocol_style = self.protocol_style_emb(protocol_id.long())   # [B, protocol_n_features]
        style = self.style_affine(protocol_style)                      # [B, 2*d_large]

        x = self.adain(x, style)    # [B, T, d_large]
        x = x.transpose(1, 2)   # [B, d_large, T]
        x = self.prelu(x)
        x = x.transpose(1, 2)   # [B, T, d_large]
        x = self.fc2(x)             # [B, T, d_dynamic_n_features]

        return x


class InputEmbedding(nn.Module):
    def __init__(
        self,
        normalization_cfg: neural_networks.InputNormalizationConfig,
        dynamic_input_size,
        metadata_input_size,
        d_model,
        protocol_n_features,
        metadata_n_features,
        num_protocols
    ):
        super().__init__()

        self.normalization = InputNormalization(normalization_cfg)

        self.protocol_n_features = protocol_n_features
        self.metadata_n_features = metadata_n_features
        self.dynamic_n_features = d_model - protocol_n_features - metadata_n_features

        assert self.dynamic_n_features > 0, "d_model too small for protocol + metadata"

        # Metadata projection
        self.metadata_linear = nn.Linear(metadata_input_size, self.metadata_n_features)

        # Protocol embedding for final concatenation
        self.protocol_emb = nn.Embedding(num_protocols, self.protocol_n_features)

        # Shared dynamic encoder
        self.dynamic_encoder = DynamicEncoder(
            dynamic_input_size=dynamic_input_size,
            d_dynamic_n_features=self.dynamic_n_features,
            protocol_n_features=self.protocol_n_features,
            num_protocols=num_protocols
        )

    def forward(self, dynamic_data, metadata, protocol_id):
        """
        dynamic_data: [B, T, 180]
        metadata:     [B, 2]
        protocol_id:  [B]

        returns:      [B, T, d_model]
        """
        B, T, _ = dynamic_data.shape

        input_tensor, input_meta_tensor = self.normalization(dynamic_data, metadata)

        # ---- dynamic data projection ----
        dyn_emb = self.dynamic_encoder(input_tensor, protocol_id)   # [B, T, dynamic_n_features]

        # ---- metadata projection ----
        meta_emb = self.metadata_linear(input_meta_tensor)          # [B, metadata_n_features]
        meta_emb = meta_emb.unsqueeze(1).expand(-1, T, -1)          # [B, T, metadata_n_features]

        # ---- protocol embedding ----
        proto_emb = self.protocol_emb(protocol_id.long())           # [B, protocol_n_features]
        proto_emb = proto_emb.unsqueeze(1).expand(-1, T, -1)        # [B, T, protocol_n_features]

        # ---- concatenate along feature dimension ----
        x = torch.cat([proto_emb, dyn_emb, meta_emb], dim=-1)       # [B, T, d_model]

        return x
'''