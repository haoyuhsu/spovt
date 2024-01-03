import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from transformer_layers import TransformerDecoderLayer, CustomTransformerDecoderLayer
from embedding import Concat_Embeddings, Add_Embeddings, ConcatBox_Embeddings
from encoder import TransformerEncoder, LabelWiseTransformerEncoder, LabelWiseCrossTransformerEncoder
from utils import generate_unit_ball, random_generate_unit_ball


class Decoder(nn.Module):
    """
    Base decoder class
    """

    @property
    def output_size(self):
        """
        Return the output size (size of the target vocabulary)
        :return:
        """
        return self._output_size


class TransformerDecoder(Decoder):
    """
    A transformer decoder with N masked layers.
    Decoder layers are masked so that an attention head cannot see the future.
    """

    def __init__(self,
                 hidden_size: int = 768,
                 ff_size: int = 2048,
                 num_layers: int = 6,
                 num_heads: int = 8, 
                 dropout: float = 0.1,
                 emb_dropout: float = 0.1,
                 freeze: bool = False,
                 src_trg_att: bool = True,
                 **kwargs):
        """
        Initialize a Transformer decoder.
        :param num_layers: number of Transformer layers
        :param num_heads: number of heads for each layer
        :param hidden_size: hidden size
        :param ff_size: position-wise feed-forward size
        :param dropout: dropout probability (1-keep)
        :param emb_dropout: dropout probability for embeddings
        :param vocab_size: size of the output vocabulary
        :param freeze: set to True keep all decoder parameters fixed
        :param kwargs:
        """
        super(TransformerDecoder, self).__init__()

        self._hidden_size = hidden_size

        # create num_layers decoder layers and put them in a list
        self.layers = nn.ModuleList([TransformerDecoderLayer(
                size=hidden_size, ff_size=ff_size, num_heads=num_heads,
                dropout=dropout,src_trg_att=src_trg_att) for _ in range(num_layers)])

        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)

        self.emb_dropout = nn.Dropout(p=emb_dropout)

    def forward(self,
                trg_embed: Tensor = None,
                encoder_output: Tensor = None,
                encoder_hidden: Tensor = None,
                src_mask: Tensor = None,
                unroll_steps: int = None,
                hidden: Tensor = None,
                trg_mask: Tensor = None,
                **kwargs):
        """
        Transformer decoder forward pass.
        :param trg_embed: embedded targets
        :param encoder_output: source representations
        :param encoder_hidden: unused
        :param src_mask:
        :param unroll_steps: unused
        :param hidden: unused
        :param trg_mask: to mask out target paddings
                         Note that a subsequent mask is applied here.
        :param kwargs:
        :return:
        """
        assert trg_mask is not None, "trg_mask required for Transformer"

        x = trg_embed

        trg_mask = trg_mask & self.subsequent_mask(
            trg_embed.size(1)).type_as(trg_mask)

        for layer in self.layers:
            x = layer(x=x, memory=encoder_output,
                      src_mask=src_mask, trg_mask=trg_mask)

        x = self.layer_norm(x)
        return x

    def __repr__(self):
        return "%s(num_layers=%r, num_heads=%r)" % (
            self.__class__.__name__, len(self.layers),
            self.layers[0].trg_trg_att.num_heads)

    def subsequent_mask(self, size: int) -> Tensor:
        """
        Mask out subsequent positions (to prevent attending to future positions)
        Transformer helper function.
        :param size: size of mask (2nd and 3rd dim)
        :return: Tensor with 0s and 1s of shape (1, size, size)
        """
        mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
        return torch.from_numpy(mask) == 0


class CustomTransformerDecoder(Decoder):
    """
    A transformer decoder with N masked layers.
    Decoder layers are masked so that an attention head cannot see the future.
    """

    def __init__(self,
                 hidden_size: int = 768,
                 hidden_bb_size: int = 64,
                 ff_size: int = 2048,
                 num_layers: int = 6,
                 num_heads: int = 8, 
                 dropout: float = 0.1,
                 emb_dropout: float = 0.1,
                 freeze: bool = False,
                 **kwargs):
        """
        Initialize a Transformer decoder.
        :param num_layers: number of Transformer layers
        :param num_heads: number of heads for each layer
        :param hidden_size: hidden size
        :param ff_size: position-wise feed-forward size
        :param dropout: dropout probability (1-keep)
        :param emb_dropout: dropout probability for embeddings
        :param vocab_size: size of the output vocabulary
        :param freeze: set to True keep all decoder parameters fixed
        :param kwargs:
        """
        super(CustomTransformerDecoder, self).__init__()

        self._hidden_size = hidden_size
        
        # create num_layers decoder layers and put them in a list
        self.layers = nn.ModuleList([CustomTransformerDecoderLayer(
                size=hidden_size, bb_size=hidden_bb_size, ff_size=ff_size, 
                num_heads=num_heads,
                dropout=dropout) for _ in range(num_layers)])

        self.layer_norm = nn.LayerNorm(hidden_size+hidden_bb_size, eps=1e-6)

        self.emb_dropout = nn.Dropout(p=emb_dropout)

    def forward(self,
                trg_embed_0: Tensor = None,
                trg_embed_1: Tensor = None,
                encoder_output: Tensor = None,
                encoder_hidden: Tensor = None,
                src_mask: Tensor = None,
                unroll_steps: int = None,
                hidden: Tensor = None,
                trg_mask: Tensor = None,
                **kwargs):
        """
        Transformer decoder forward pass.
        :param trg_embed: embedded targets
        :param encoder_output: source representations
        :param encoder_hidden: unused
        :param src_mask:
        :param unroll_steps: unused
        :param hidden: unused
        :param trg_mask: to mask out target paddings
                         Note that a subsequent mask is applied here.
        :param kwargs:
        :return:
        """
        assert trg_mask is not None, "trg_mask required for Transformer"

        x_0 = trg_embed_0 # spatial
        x_1 = trg_embed_1 # semantic

        trg_mask = trg_mask & self.subsequent_mask(
            trg_embed_0.size(1)).type_as(trg_mask)

        for layer in self.layers:
            x = layer(spatial_x=x_0, semantic_x=x_1, memory=encoder_output,
                      src_mask=src_mask, trg_mask=trg_mask)

        x = self.layer_norm(x)
        return x

    def __repr__(self):
        return "%s(num_layers=%r, num_heads=%r)" % (
            self.__class__.__name__, len(self.layers),
            self.layers[0].trg_trg_att.num_heads)

    def subsequent_mask(self, size: int) -> Tensor:
        """
        Mask out subsequent positions (to prevent attending to future positions)
        Transformer helper function.
        :param size: size of mask (2nd and 3rd dim)
        :return: Tensor with 0s and 1s of shape (1, size, size)
        """
        mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
        return torch.from_numpy(mask) == 0


class PointDecoder(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """
    def __init__(self, hidden_size=256, num_layers=4, attn_heads=8, dropout=0.0, seg_class = 4):
        
        super(PointDecoder, self).__init__()

        self.hidden_size = hidden_size
        self.embed = nn.Linear(259,256)
        self.decoder = TransformerEncoder(hidden_size=hidden_size, ff_size=1024, 
                                          num_layers=num_layers, \
                                          num_heads=attn_heads, dropout=dropout, emb_dropout=dropout)
        self.final_gen_point = nn.Linear(256,3)
        self.final_gen_seg = nn.Linear(256,4)
        self.final_block = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128, elementwise_affine=False),
            nn.ReLU(inplace=True),
            nn.Linear(128,32),
            nn.LayerNorm(32, elementwise_affine=False),
            nn.ReLU(inplace=True),
            nn.Linear(32,3)
        )

    def forward(self, encoder_output):
        bs, part = encoder_output.size()[0], encoder_output.size()[1]
        part_feat_cat = encoder_output.mean(1).view(bs,1,-1).repeat(1,1024,1)
        balls = torch.from_numpy(generate_unit_ball(1024, scale=0.5)).repeat(bs,1,1).cuda().float()
        balls = torch.cat((balls, part_feat_cat),2)
        balls = self.embed(balls)
        balls = self.decoder(balls, None)
        balls = self.final_block(balls)
        return balls


class PointDecoder2(nn.Module):
    """
    NAT decoder composed of transformer encoder
    """
    def __init__(self, hidden_size=256, num_layers=4, attn_heads=8, dropout=0.3, seg_class = 4, max_token=6):
        
        super(PointDecoder2, self).__init__()
        self.seg_class = seg_class
        self.max_token = max_token
        self.hidden_size = hidden_size
        self.embed = nn.Linear(261,256)
        self.decoder = LabelWiseTransformerEncoder(hidden_size=hidden_size, ff_size=1024, 
                                          num_layers=num_layers, \
            num_heads=attn_heads, dropout=dropout, max_label = max_token-2, emb_dropout=dropout)
        self.final_block = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128, elementwise_affine=False),
            nn.ReLU(inplace=True),
            nn.Linear(128,32),
            nn.LayerNorm(32, elementwise_affine=False),
            nn.ReLU(inplace=True),
            nn.Linear(32,3)
        )

    def forward(self, encoder_output, input_point, input_label):
        bs, part = encoder_output.size()[0], encoder_output.size()[1]
        point_num = input_point.size(1)
        pad_mask = 1 - (input_label == -2).long()
        reliable_token = F.one_hot(pad_mask, num_classes=2)
        point_pad_mask = pad_mask.unsqueeze(2).repeat(1,1,3)
        for i in range(bs):
            label_candidate = torch.unique(input_label[i]).clone().cpu().numpy()
            if -2 in label_candidate:
                label_candidate = np.delete(label_candidate,np.where(label_candidate==-2))
            label = torch.tensor(np.random.choice(label_candidate, point_num), device = input_point.device)
            input_label[i] = input_label[i] * pad_mask[i] + label * (1-pad_mask[i])
        balls = torch.from_numpy(generate_unit_ball(point_num, scale=0.5)).repeat(bs,1,1).cuda().float()
        input_point = input_point * point_pad_mask + balls * (1-point_pad_mask)
        id_0 = torch.arange(bs).view(-1,1)
        part_feat_cat = encoder_output[id_0,input_label]
        balls = torch.cat((input_point, part_feat_cat, reliable_token),2)
        balls = self.embed(balls)
        balls = self.decoder(balls,input_label ,None)
        balls = self.final_block(balls)
        return (input_point + balls)[:,self.max_token-2:,:], input_label[:,self.max_token-2:]


class PointDecoder3(nn.Module):
    """
    NAT decoder composed of transformer encoder, point num of each part predicted by encoder
    """
    def __init__(self, hidden_size=256, num_layers=4, attn_heads=8, dropout=0.3, seg_class = 4, max_token=6, point_num = 1024):
        
        super(PointDecoder3, self).__init__()
        self.seg_class = seg_class
        self.max_token = max_token
        self.hidden_size = hidden_size
        self.point_num = point_num
        self.embed = nn.Linear(261,256)
        self.decoder = LabelWiseTransformerEncoder(hidden_size=hidden_size, ff_size=1024, 
                                          num_layers=num_layers, \
            num_heads=attn_heads, dropout=dropout, max_label = max_token-2, emb_dropout=dropout)
        self.final_block = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128, elementwise_affine=False),
            nn.ReLU(inplace=True),
            nn.Linear(128,32),
            nn.LayerNorm(32, elementwise_affine=False),
            nn.ReLU(inplace=True),
            nn.Linear(32,3)
        )

    def forward(self, encoder_output, input_point, input_label, input_ratio, point_ratio, n):
        bs, part = encoder_output.size()[0], encoder_output.size()[1]
        n -= (self.max_token-2)
        pad_label_nums = torch.round(point_ratio*self.point_num - input_ratio*n.unsqueeze(1).repeat(1,part))
        part_points = torch.round(point_ratio*self.point_num)
        point_pad_mask = torch.zeros(bs, self.point_num, device = input_ratio.device)
        balls = torch.from_numpy(generate_unit_ball(self.point_num, scale=0.5)).repeat(bs,1,1).cuda().float()
        labels = torch.zeros(bs, self.point_num, device = input_ratio.device)
        for i in range(bs):
            cnt = 0
            for j in range(part):
                part_select = min([part_points[i,j], part_points[i,j]-pad_label_nums[i,j]]).long()
                if part_select != 0:
                    label_mask = torch.nonzero((input_label[i]==j).long()).squeeze().reshape(-1)
                    selected_points = input_point[i, label_mask]
                    indexs = torch.randperm(len(label_mask))
                    selected_points = selected_points[indexs][:part_select]
                    part_n = len(selected_points)
                    balls[i, cnt:cnt+part_n,:] = selected_points
                    labels[i, cnt:cnt+part_n] = j 
                    point_pad_mask[i, cnt:cnt+part_n] = 1
                    cnt += part_n
            prob = torch.maximum(pad_label_nums[i],torch.zeros(part, device=pad_label_nums.device))
            if (cnt < self.point_num) and (prob.sum() >= 1):
                labels[i, cnt:] = torch.multinomial(prob, self.point_num-cnt, replacement=True)        
        reliable_token = F.one_hot(point_pad_mask.long(), num_classes=2)
        id_0 = torch.arange(bs).view(-1,1)
        part_feat_cat = encoder_output[id_0,labels.long()]
        input_point = torch.cat((balls, part_feat_cat, reliable_token),2)
        input_point = self.embed(input_point)
        input_point = self.decoder(input_point,labels ,None)
        input_point = self.final_block(input_point)
        return (input_point + balls), labels


class PointDecoder4(nn.Module):
    """
    NAT decoder composed of transformer encoder, point num of each part predicted by encoder
    """
    def __init__(self, hidden_size=256, num_layers=4, attn_heads=8, dropout=0.3, seg_class = 4, max_token=6, point_num = 1024):        
        super(PointDecoder4, self).__init__()
        self.seg_class = seg_class
        self.max_token = max_token
        self.hidden_size = hidden_size
        self.point_num = point_num
        self.embed = nn.Linear(261,256)
        self.decoder = LabelWiseCrossTransformerEncoder(hidden_size=hidden_size, ff_size=1024, 
                                          num_layers=num_layers, \
            num_heads=attn_heads, dropout=dropout, max_label = max_token-2, emb_dropout=dropout)
        template_origin = torch.Tensor(np.load('../utils/chair_sampled.npy')[:,:3])
        self.template = [template_origin[:1024],
                        template_origin[1024:2048],
                        template_origin[2048:3072],
                        template_origin[3072:4096]]
        self.template_len = [len(self.template[i]) for i in range(len(self.template))]
        self.final_block = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128, elementwise_affine=False),
            nn.ReLU(inplace=True),
            nn.Linear(128,32),
            nn.LayerNorm(32, elementwise_affine=False),
            nn.ReLU(inplace=True),
            nn.Linear(32,3)
        )

    def forward(self, encoder_output, input_point, input_label, input_ratio, point_ratio, n, encoder_output_full=None, mask=None):
        bs, part = encoder_output.size()[0], encoder_output.size()[1]
        pad_label_nums = torch.round(point_ratio*self.point_num - input_ratio*n.unsqueeze(1).repeat(1,part))
        part_points = torch.round(point_ratio*self.point_num)
        point_pad_mask = torch.zeros(bs, self.point_num, device = input_ratio.device)
        balls = torch.zeros((bs, self.point_num, 3), device = input_ratio.device)
        labels = torch.zeros(bs, self.point_num, device = input_ratio.device)
        for i in range(bs):
            cnt = 0
            prob = torch.maximum(pad_label_nums[i],torch.zeros(part, device=pad_label_nums.device))
            part_select_obj = torch.minimum(part_points[i], part_points[i]-pad_label_nums[i]).long()
            if (part_select_obj.sum() < self.point_num) and (prob.sum() >= 1):
                labels_pad = torch.multinomial(prob, self.point_num-part_select_obj.sum(), replacement=True)
            else:
                labels_pad = None
            for j in range(part):
                part_select = part_select_obj[j]
                if part_select != 0:
                    label_mask = torch.nonzero((input_label[i]==j).long()).squeeze().reshape(-1)
                    if labels_pad != None:
                        pad_num = len(torch.nonzero((labels_pad==j).long()).squeeze().reshape(-1))
                    else:
                        pad_num = 0
                    selected_points = input_point[i, label_mask]
                    indexs = torch.randperm(len(label_mask))
                    selected_points = selected_points[indexs][:part_select]
                    start_index = 0
                    pad_points = (self.template[j][start_index:start_index+pad_num]).to(selected_points.device)
                    part_n = len(selected_points) + len(pad_points)
                    balls[i, cnt:cnt+part_n,:] = torch.cat((selected_points,pad_points), 0)
                    labels[i, cnt:cnt+part_n] = j 
                    point_pad_mask[i, cnt:cnt+len(selected_points)] = 1
                    cnt += part_n
                else:
                    label_mask = torch.nonzero((input_label[i]==j).long()).squeeze().reshape(-1)
                    if labels_pad != None:
                        pad_num = len(torch.nonzero((labels_pad==j).long()).squeeze().reshape(-1))
                    else:
                        pad_num = 0
                    selected_points = input_point[i, label_mask]
                    indexs = torch.randperm(len(label_mask))
                    selected_points = selected_points[indexs][:part_select]
                    start_index = 0
                    pad_points = (self.template[j][start_index:start_index+pad_num]).to(selected_points.device)
                    part_n = len(selected_points) + len(pad_points)
                    balls[i, cnt:cnt+part_n,:] = torch.cat((selected_points,pad_points), 0)
                    labels[i, cnt:cnt+part_n] = j 
                    point_pad_mask[i, cnt:cnt+len(selected_points)] = 1 
                    cnt += part_n
        reliable_token = F.one_hot(point_pad_mask.long(), num_classes=2)
        id_0 = torch.arange(bs).view(-1,1)
        part_feat_cat = encoder_output[id_0,labels.long()]
        input_point = torch.cat((balls, part_feat_cat, reliable_token),2)
        input_point = self.embed(input_point)
        input_point = self.decoder(input_point,labels ,None, encoder_output_full[:,part:,:], mask[:,:,part:])
        input_point = self.final_block(input_point)
        return (input_point + balls), labels


class TVAEPointDecoder(nn.Module):
    """
    NAT decoder composed of transformer encoder, point num of each part predicted by encoder
    """
    def __init__(self, hidden_size=256, num_layers=4, attn_heads=8, dropout=0.3, seg_class = 4, max_token=6, point_num = 256):
        super(TVAEPointDecoder, self).__init__()
        self.seg_class = seg_class
        self.max_token = max_token
        self.hidden_size = hidden_size
        self.point_num = point_num
        self.decoder = LabelWiseCrossTransformerEncoder(hidden_size=hidden_size, ff_size=1024, 
                                          num_layers=num_layers, \
            num_heads=attn_heads, dropout=dropout, max_label = max_token-2, emb_dropout=dropout)
        self.final_block = nn.Sequential(
            nn.Linear(self.hidden_size, 128),
            nn.LayerNorm(128, elementwise_affine=False),
            nn.ReLU(inplace=False),
            nn.Linear(128,32),
            nn.LayerNorm(32, elementwise_affine=False),
            nn.ReLU(inplace=False),
            nn.Linear(32,3)
        )

    def forward(self, z, out_label, input_ratio, encoder_output, mask=None, self_recon = False):
        bs, part = encoder_output.size()[0], self.max_token-2
        if self_recon:
            point = self.decoder(z,out_label ,mask[:,:,part:], encoder_output, mask[:,:,part:])
        else:
            point = self.decoder(z,out_label ,None, encoder_output, mask[:,:,part:])
        point = self.final_block(point)
        return point


class TVAEPointRefineDecoder(nn.Module):
    """
    NAT decoder composed of transformer encoder, point num of each part predicted by encoder
    """
    def __init__(self, hidden_size=256, num_layers=4, attn_heads=8, dropout=0.3, seg_class = 4, max_token=6, point_num = 256, expand = 4):
        super(TVAEPointRefineDecoder, self).__init__()
        self.seg_class = seg_class
        self.max_token = max_token
        self.hidden_size = hidden_size
        self.point_num = point_num
        self.expand = expand
        self.decoder = LabelWiseCrossTransformerEncoder(hidden_size=hidden_size, ff_size=1024, 
                                          num_layers=num_layers, \
            num_heads=attn_heads, dropout=dropout, max_label = max_token-2, emb_dropout=dropout)
        self.final_block = nn.Sequential(
            nn.Linear(self.hidden_size, 128),
            nn.LayerNorm(128, elementwise_affine=False),
            nn.ReLU(inplace=False),
            nn.Linear(128,32),
            nn.LayerNorm(32, elementwise_affine=False),
            nn.ReLU(inplace=False),
            nn.Linear(32,3*expand)
        )

    def forward(self, z, out_label, input_ratio, encoder_output, mask=None, self_recon = False):
        bs, part = encoder_output.size()[0], self.max_token-2
        if self_recon:
            point = self.decoder(z,out_label ,mask[:,:,part:], encoder_output, mask[:,:,part:])
        else:
            point = self.decoder(z,out_label ,None, encoder_output, mask[:,:,part:])
        point = self.final_block(point)
        return point


class PointDecoderSimple(nn.Module):
    """
    NAT decoder composed of transformer encoder, point num of each part predicted by encoder
    """
    def __init__(self, hidden_size=256, num_layers=4, attn_heads=8, dropout=0.3, seg_class = 4, max_token=6, point_num = 1024):
        super(PointDecoderSimple, self).__init__()
        self.seg_class = seg_class
        self.max_token = max_token
        self.hidden_size = hidden_size
        self.point_num = point_num
        self.embed = nn.Linear(261,256)
        self.decoder = LabelWiseCrossTransformerEncoder(hidden_size=hidden_size, ff_size=self.hidden_size*4, 
                                          num_layers=num_layers, \
            num_heads=attn_heads, dropout=dropout, max_label = max_token-2, emb_dropout=dropout)
        self.position_encoding = nn.Embedding(16384, hidden_size, padding_idx=-1)
        self.final_block = nn.Sequential(
            nn.Linear(self.hidden_size, 128),
            nn.LayerNorm(128, elementwise_affine=False),
            nn.ReLU(inplace=True),
            nn.Linear(128,32),
            nn.LayerNorm(32, elementwise_affine=False),
            nn.ReLU(inplace=True),
            nn.Linear(32,3)
        )

    def forward(self, encoder_output, input_point, input_label, input_ratio, point_ratio, n, encoder_output_full=None, mask=None):
        bs, part = encoder_output.size()[0], encoder_output.size()[1]
        part_points = torch.ceil(point_ratio*self.point_num).int()
        position = torch.zeros((bs, self.point_num)).to(encoder_output.device)
        labels = torch.zeros((bs, self.point_num)).to(encoder_output.device).long()
        for i in range(bs):
            pointer = 0
            for j in range(self.max_token-2):
                if pointer + part_points[i,j] <= self.point_num:
                    position[i, pointer:pointer+part_points[i,j]] = torch.arange(part_points[i,j])
                    labels[i, pointer:pointer+part_points[i,j]] = j
                    pointer += part_points[i,j]
                else:
                    part_length = self.point_num-pointer
                    position[i, pointer:] = torch.arange(part_length)
                    labels[i, pointer:] = j
                    pointer += part_length
        id_0 = torch.arange(bs).view(-1,1)
        part_feat_cat = encoder_output[id_0,labels]
        position_encodings = self.position_encoding(position.long())
        input_point = position_encodings + part_feat_cat
        input_point = self.decoder(input_point,labels ,None, encoder_output_full[:,part:,:], mask[:,:,part:])
        input_point = self.final_block(input_point)
        return (input_point), labels