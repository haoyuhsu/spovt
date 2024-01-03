import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformer_layers import TransformerEncoderLayer, LabelWiseTransformerEncoderLayer
from embedding import Point_Embeddings, Fuse_Point_Embeddings, Sentence_Embeddings


class Encoder(nn.Module):
    """
    Base encoder class
    """
    @property
    def output_size(self):
        """
        Return the output size
        :return:
        """
        return self._output_size


class TransformerEncoder(Encoder):
    """
    Transformer Encoder
    """
    #pylint: disable=unused-argument
    def __init__(self,
                 hidden_size: int = 512,
                 ff_size: int = 2048,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 emb_dropout: float = 0.1,
                 freeze: bool = False,
                 **kwargs):
        """
        Initializes the Transformer.
        :param hidden_size: hidden size and size of embeddings
        :param ff_size: position-wise feed-forward layer size.
          (Typically this is 2*hidden_size.)
        :param num_layers: number of layers
        :param num_heads: number of heads for multi-headed attention
        :param dropout: dropout probability for Transformer layers
        :param emb_dropout: Is applied to the input (word embeddings).
        :param freeze: freeze the parameters of the encoder during training
        :param kwargs:
        """
        super(TransformerEncoder, self).__init__()

        # build all (num_layers) layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(size=hidden_size, ff_size=ff_size,
                                    num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)])

        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.emb_dropout = nn.Dropout(p=emb_dropout)

        self._output_size = hidden_size
        self._hidden_size = hidden_size

    #pylint: disable=arguments-differ
    def forward(self,
                embed_src: Tensor,
                mask: Tensor) -> (Tensor, Tensor):
        """
        Pass the input (and mask) through each layer in turn.
        Applies a Transformer encoder to sequence of embeddings x.
        The input mini-batch x needs to be sorted by src length.
        x and mask should have the same dimensions [batch, time, dim].
        :param embed_src: embedded src inputs,
            shape (batch_size, src_len, embed_size)
        :param mask: indicates padding areas (zeros where padding), shape
            (batch_size, src_len, embed_size)
        :return:
            - output: hidden states with
                shape (batch_size, max_length, directions*hidden),
            - hidden_concat: last hidden state with
                shape (batch_size, directions*hidden)
        """
        x = embed_src

        for layer in self.layers:
            x = F.relu(layer(x, mask), inplace= False)
            
        return self.layer_norm(x)

    def __repr__(self):
        return "%s(num_layers=%r, num_heads=%r)" % (
            self.__class__.__name__, len(self.layers),
            self.layers[0].src_src_att.num_heads)


class LabelWiseTransformerEncoder(Encoder):
    """
    Transformer Encoder
    """
    #pylint: disable=unused-argument
    def __init__(self,
                 hidden_size: int = 512,
                 ff_size: int = 2048,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 max_label: int = 4,
                 emb_dropout: float = 0.1,
                 freeze: bool = False,
                 **kwargs):
        """
        Initializes the Transformer.
        :param hidden_size: hidden size and size of embeddings
        :param ff_size: position-wise feed-forward layer size.
          (Typically this is 2*hidden_size.)
        :param num_layers: number of layers
        :param num_heads: number of heads for multi-headed attention
        :param dropout: dropout probability for Transformer layers
        :param emb_dropout: Is applied to the input (word embeddings).
        :param freeze: freeze the parameters of the encoder during training
        :param kwargs:
        """
        super(LabelWiseTransformerEncoder, self).__init__()

        # build all (num_layers) layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(size=hidden_size, ff_size=ff_size,
                                    num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)])
        self.layers.append(LabelWiseTransformerEncoderLayer(size=hidden_size, ff_size=ff_size,
                                    num_heads=num_heads, dropout=dropout, max_label = max_label))

        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.emb_dropout = nn.Dropout(p=emb_dropout)

        self._output_size = hidden_size
        self._hidden_size = hidden_size

    #pylint: disable=arguments-differ
    def forward(self,
                embed_src: Tensor,
                label: Tensor,
                mask: Tensor) -> (Tensor, Tensor):
        """
        Pass the input (and mask) through each layer in turn.
        Applies a Transformer encoder to sequence of embeddings x.
        The input mini-batch x needs to be sorted by src length.
        x and mask should have the same dimensions [batch, time, dim].
        :param embed_src: embedded src inputs,
            shape (batch_size, src_len, embed_size)
        :param mask: indicates padding areas (zeros where padding), shape
            (batch_size, src_len, embed_size)
        :return:
            - output: hidden states with
                shape (batch_size, max_length, directions*hidden),
            - hidden_concat: last hidden state with
                shape (batch_size, directions*hidden)
        """
        x = embed_src

        for layer in self.layers[:-1]:
            x = F.relu(layer(x, mask), inplace= False)
        x = F.relu(self.layers[-1](x, label, mask), inplace= False)
            
        return self.layer_norm(x)

    def __repr__(self):
        return "%s(num_layers=%r, num_heads=%r)" % (
            self.__class__.__name__, len(self.layers),
            self.layers[0].src_src_att.num_heads)


class LabelWiseCrossTransformerEncoder(Encoder):
    """
    Transformer Encoder
    """
    #pylint: disable=unused-argument
    def __init__(self,
                 hidden_size: int = 512,
                 ff_size: int = 2048,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 max_label: int = 4,
                 emb_dropout: float = 0.1,
                 freeze: bool = False,
                 **kwargs):
        """
        Initializes the Transformer.
        :param hidden_size: hidden size and size of embeddings
        :param ff_size: position-wise feed-forward layer size.
          (Typically this is 2*hidden_size.)
        :param num_layers: number of layers
        :param num_heads: number of heads for multi-headed attention
        :param dropout: dropout probability for Transformer layers
        :param emb_dropout: Is applied to the input (word embeddings).
        :param freeze: freeze the parameters of the encoder during training
        :param kwargs:
        """
        super(LabelWiseCrossTransformerEncoder, self).__init__()

        # build all (num_layers) layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(size=hidden_size, ff_size=ff_size,
                                    num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)])
        self.layers.append(LabelWiseTransformerEncoderLayer(size=hidden_size, ff_size=ff_size,
                                    num_heads=num_heads, dropout=dropout, max_label = max_label))

        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.emb_dropout = nn.Dropout(p=emb_dropout)

        self._output_size = hidden_size
        self._hidden_size = hidden_size

    #pylint: disable=arguments-differ
    def forward(self,
                embed_src: Tensor,
                label: Tensor,
                self_mask: Tensor,
                cross_embed:Tensor,
                cross_mask:Tensor,) -> (Tensor, Tensor):
        """
        Pass the input (and mask) through each layer in turn.
        Applies a Transformer encoder to sequence of embeddings x.
        The input mini-batch x needs to be sorted by src length.
        x and mask should have the same dimensions [batch, time, dim].
        :param embed_src: embedded src inputs,
            shape (batch_size, src_len, embed_size)
        :param mask: indicates padding areas (zeros where padding), shape
            (batch_size, src_len, embed_size)
        :return:
            - output: hidden states with
                shape (batch_size, max_length, directions*hidden),
            - hidden_concat: last hidden state with
                shape (batch_size, directions*hidden)
        """
        x = embed_src
        cnt = 0
        for layer in self.layers[:-1]:
            if cnt == 1:
                x = F.relu(layer(x,self_mask,cross_embed,cross_mask))
            else:
                x = F.relu(layer(x, self_mask), inplace= False)
            cnt+=1
        x = F.relu(self.layers[-1](x, label, self_mask), inplace= False)
            
        return self.layer_norm(x)

    def __repr__(self):
        return "%s(num_layers=%r, num_heads=%r)" % (
            self.__class__.__name__, len(self.layers),
            self.layers[0].src_src_att.num_heads)


class RelEncoder(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """
    def __init__(self, vocab_size=204, obj_classes_size=154, hidden_size=512, num_layers=6, attn_heads=8, dropout=0.1, cfg=None):
        super(RelEncoder, self).__init__()

        self.input_embeddings = Sentence_Embeddings(vocab_size, obj_classes_size, hidden_size, max_rel_pair= 33)
        
        self.encoder = TransformerEncoder(hidden_size=hidden_size, ff_size=hidden_size * 4, 
                                          num_layers=num_layers, \
            num_heads=attn_heads, dropout=dropout, emb_dropout=dropout)
        self.hidden_size = hidden_size

        self.vocab_classifier = nn.Linear(hidden_size, vocab_size)
        self.obj_id_classifier = nn.Linear(hidden_size, obj_classes_size)
        self.token_type_classifier = nn.Linear(hidden_size, 4)

    def forward(self, input_token, input_obj_id, segment_label, token_type, src_mask):
        batch_size = input_token.shape[0]
        
        src, class_embeds = self.input_embeddings(input_token, 
                                        input_obj_id, segment_label, token_type)

        encoder_output = self.encoder(src, src_mask)

        vocab_logits = self.vocab_classifier(encoder_output)
        obj_id_logits = self.obj_id_classifier(encoder_output)
        token_type_logits = self.token_type_classifier(encoder_output)

        return encoder_output, vocab_logits, obj_id_logits, token_type_logits, src, class_embeds


class PointEncoder(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """
    def __init__(self, point_dim=3, max_token=6, hidden_size=256, num_layers=6, attn_heads=8, dropout=0.3, cfg=None):
        super(PointEncoder, self).__init__()

        self.input_embeddings = Fuse_Point_Embeddings(point_dim, hidden_size, max_token= 6, hidden_dropout_prob=dropout)
        
        self.encoder = LabelWiseTransformerEncoder(hidden_size=hidden_size, ff_size=hidden_size * 4, 
                                          num_layers=num_layers, \
            num_heads=attn_heads, dropout=dropout, max_label = max_token-2, emb_dropout=dropout)
        self.hidden_size = hidden_size

        self.label_classifier = nn.Linear(hidden_size, max_token-2)
        self.ratio_regressor = Ratio_Regressor(hidden_size)
        self.max_token = max_token

    def forward(self, input_points, token_type, label,  src_mask, pretrain = False, input_ratio = None):
        batch_size = input_points.shape[0]
        src, class_embeds = self.input_embeddings(input_points,  token_type)
        encoder_output = self.encoder(src, label ,src_mask)
        if pretrain:
            label_logits = self.label_classifier(encoder_output)
            ratio_logits = self.ratio_regressor(encoder_output[:, :self.max_token-2], input_ratio)
            return encoder_output, label_logits, ratio_logits, None, None
        else:
            ratio_logits = self.ratio_regressor(encoder_output[:, :self.max_token-2], input_ratio)
            return encoder_output, None, ratio_logits, None, None
        

class Ratio_Regressor(nn.Module):
    def __init__(self, input_dim, num_parts = 4):
        super(Ratio_Regressor, self).__init__()
        self.num_parts = num_parts
        self.ratio_embed = nn.Linear(num_parts, input_dim)
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(inplace=False),
            nn.LayerNorm(64, elementwise_affine=False),
            nn.Dropout(0.5),
            nn.Linear(64,1)
        )

    def forward(self, encoder_output, input_ratio):
        batch_size = encoder_output.shape[0]
        x = encoder_output + self.ratio_embed(input_ratio).unsqueeze(1).repeat(1,self.num_parts,1)
        x = F.softmax(self.regressor(x).squeeze(), -1)
        return x


class TVAEPointEncoder(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """
    def __init__(self, point_dim=3, max_token=6, hidden_size=256, num_layers=6, attn_heads=8, dropout=0.3, cfg=None):
        super(TVAEPointEncoder, self).__init__()
        self.input_embeddings = Fuse_Point_Embeddings(point_dim, hidden_size, max_token= max_token, hidden_dropout_prob=dropout)
        self.encoder = LabelWiseTransformerEncoder(hidden_size=hidden_size, ff_size=hidden_size * 4, 
                                          num_layers=num_layers, \
            num_heads=attn_heads, dropout=dropout, max_label = max_token-2, emb_dropout=dropout)
        self.hidden_size = hidden_size
        self.label_classifier = nn.Linear(hidden_size, max_token-2)
        self.ratio_regressor = Ratio_Regressor(hidden_size, num_parts=max_token-2)
        self.max_token = max_token
        self.mu_mlp = nn.ModuleList(
            [
                nn.Linear(hidden_size, hidden_size) for _ in range(self.max_token-2)
            ]
        )
        self.std_mlp = nn.ModuleList(
            [
                nn.Linear(hidden_size, hidden_size) for _ in range(self.max_token-2)
            ]
        )
        self.l_norm = nn.LayerNorm(self.hidden_size, elementwise_affine=False)
        self.hidden_size = hidden_size

    def forward(self, input_points, token_type, label,  src_mask, pretrain = False, input_ratio = None):
        batch_size = input_points.shape[0]
        src, class_embeds = self.input_embeddings(input_points,  token_type)
        encoder_output = self.encoder(src, label ,src_mask)
        mu = torch.zeros_like(encoder_output)
        log_std = torch.zeros_like(encoder_output)
        for i in range(self.max_token-2):
            ### mu_i only consider output related to i-th part
            mu_i = self.mu_mlp[i](encoder_output[:, self.max_token-2:])*((label==i).int().unsqueeze(-1).repeat(1,1,self.hidden_size)[:, self.max_token-2:].to(encoder_output.device))
            log_std_i = self.std_mlp[i](encoder_output[:, self.max_token-2:])*((label==i).int().unsqueeze(-1).repeat(1,1,self.hidden_size)[:, self.max_token-2:].to(encoder_output.device))
            mu[:, self.max_token-2:] = mu[:, self.max_token-2:] + mu_i
            log_std[:, self.max_token-2:] = log_std[:, self.max_token-2:] + log_std_i
        if pretrain:
            label_logits = self.label_classifier(encoder_output)
            ratio_logits = self.ratio_regressor(encoder_output[:, :self.max_token-2], input_ratio)
            return encoder_output, label_logits, ratio_logits, None, None
        else:
            input_ratio = torch.zeros_like(input_ratio)
            ratio_logits = self.ratio_regressor(encoder_output[:, :self.max_token-2], input_ratio)
            return encoder_output, None, ratio_logits, mu, log_std
        
    def normalize(self, labels, samples, prototypes):
        point_num = samples.size(1)
        normalized_samples = torch.zeros_like(samples)
        for i in range(self.max_token-2):
            d_samples = ((samples-prototypes[:,i].unsqueeze(1).repeat(1,point_num,1))/(torch.abs(prototypes[:,i]).reshape(-1,1,samples.size(-1)).repeat(1,point_num,1)))*(labels==i).int().to(samples.device).unsqueeze(-1).repeat(1,1,samples.size(-1))
            safe_tensor = torch.where(torch.isnan(d_samples), torch.zeros_like(d_samples), d_samples)
            d_samples = d_samples * (torch.isnan(d_samples).int())
            print(d_samples)
            normalized_samples += d_samples
        return normalized_samples


def normalize(labels, samples, prototypes):
    normalized_samples = torch.zeros_like(samples)
    for i in range(4):
        d_samples = ((samples-prototypes[:,i].unsqueeze(1).repeat(1,10,1))/(torch.norm(prototypes[:,i], dim=1)+1e-8))*(labels==i).int().to(samples.device).unsqueeze(-1).repeat(1,1,samples.size(-1))
        normalized_samples += d_samples
        print(torch.norm(prototypes[:,i], dim=1))
    return normalized_samples


if __name__ == '__main__':
    labels = torch.Tensor([[0,0,0,1,1,1,2,2,3,3]])
    samples = torch.Tensor([[
        [1,1,1],
        [1,1,1],
        [1,1,1],
        [2,2,2],
        [2,2,2],
        [2,2,2],
        [3,3,3],
        [3,3,3],
        [4,4,4],
        [4,4,4]
    ]])+1
    prototypes = torch.Tensor(
        [[
            [1,1,1],
            [2,2,2],
            [3,3,3],
            [4,4,4]
        ]]
    )
    print(labels.size(), samples.size(),prototypes.size())
    print(normalize(labels, samples, prototypes))