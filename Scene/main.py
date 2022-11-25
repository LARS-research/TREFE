import sys

sys.path.append('..')
import math
import os
import random
import string
import time
import builtins
import cv2
import lmdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import AverageMeter, get_logger
from yacs.config import CfgNode as CN
import torch.nn.parallel
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
from torchvision import transforms 
import six 
import PIL 
import warnings
from utils.scene_transforms import CVColorJitter, CVDeterioration, CVGeometry

h_map_to_channels = {32: 32, 16: 32, 8: 64, 4: 128, 2:256, 1: 512}


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class SeModule(nn.Module):
    def __init__(self, in_channels, channels, se_ratio=12):
        super(SeModule, self).__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, channels // se_ratio, kernel_size=1, padding=0),
            nn.BatchNorm2d(channels // se_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // se_ratio, channels, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(x)


class SwishImplementation(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class CompactMBV3Block_k5e6(nn.Module):
    kernel_size = 5
    ratio = 6

    '''expand + depthwise + pointwise'''
    def __init__(self, resolution, old_resolution):
        super().__init__()
        h, w = resolution 
        out_size = h_map_to_channels[h]
        expand_size = out_size * self.ratio 

        old_h, old_w = old_resolution
        old_c = h_map_to_channels[old_h]
        stride = [old_h // h, old_w // w]

        self.downsample = None 
        if stride != [1, 1]:
            self.downsample = nn.Sequential(
                nn.Conv2d(old_c, out_size, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_size)
            )

        self.conv1 = nn.Conv2d(old_c, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)

        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=self.kernel_size, stride=stride, padding=self.kernel_size//2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)

        self.se = SeModule(expand_size, expand_size)

        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)

        self.swish = Swish()

    def forward(self, x):
        identity = x
        out = self.swish(self.bn1(self.conv1(x)))
        out = self.swish(self.se(self.bn2(self.conv2(out))))
        out = self.bn3(self.conv3(out))
        if self.downsample:
            identity = self.downsample(identity)
        out += identity 
        out = F.relu(out, inplace=True)
        return out


class CompactMBV3Block_k3e6(CompactMBV3Block_k5e6):
    kernel_size = 3
    ratio = 6


class CompactMBV3Block_k3e1(CompactMBV3Block_k5e6):
    kernel_size = 3
    ratio = 1


class CompactMBV3Block_k5e1(CompactMBV3Block_k5e6):
    kernel_size = 5
    ratio = 1


class RelativeSinusoidalPositionalEmbedding(nn.Module):

    def __init__(self, embedding_dim, padding_idx=0, init_size=1568):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        assert init_size%2==0
        weights = self.get_embedding(init_size+1, embedding_dim, padding_idx)
        self.register_buffer('weights', weights)

    def get_embedding(self, num_embeddings, embedding_dim, padding_idx=None):
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(-num_embeddings//2, num_embeddings//2, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        self.origin_shift = num_embeddings//2 + 1
        return emb

    def forward(self, seq_len):
        max_pos = self.padding_idx + seq_len 
        if max_pos > self.origin_shift:
            weights = self.get_embedding(
                max_pos*2,
                self.embedding_dim,
                self.padding_idx,
            )
            del self.weights
            self.origin_shift = weights.size(0)//2
            self.register_buffer('weights', weights)
        positions = torch.arange(-seq_len, seq_len).long() + self.origin_shift  # 2*seq_len
        positions = positions.to(device=self.weights.device)
        embed = self.weights.index_select(0, positions.long()).detach()
        return embed 


def calc_rel_position_embedding(seq_len, embedding_dim):
    embedder = RelativeSinusoidalPositionalEmbedding(embedding_dim)
    embedding = embedder(seq_len)
    return embedding 


class RelativeMultiHeadAttnPro(nn.Module):
    def __init__(self, d_model, n_head, dropout, seq_len, add_scale=True, add_rel_pos=True):
        super().__init__()
        self.qkv_linear = nn.Linear(d_model, d_model*3, bias=False)
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.dropout = nn.Dropout(dropout)
        self.seq_len = seq_len 

        self.add_scale = add_scale
        self.add_rel_pos = add_rel_pos
        if add_rel_pos:
            self.r_r_bias = nn.Parameter(nn.init.xavier_normal_(
                torch.zeros(1, n_head, 1, self.head_dim))) # v 
            self.r_w_bias = nn.Parameter(nn.init.xavier_normal_(
                torch.zeros(1, n_head, 1, self.head_dim))) # u
            self.register_buffer(
                'rel_pos_embedding', 
                calc_rel_position_embedding(seq_len=seq_len, embedding_dim=self.head_dim).T)  # d,2t

    def forward(self, x, prev, skip_op):
        """
        x: (b, T, c)
        """
        batch_size, max_len, d_model = x.size() 
        qkv = self.qkv_linear(x)
        q, k, v = torch.chunk(qkv, chunks=3, dim=-1)

        q = q.view(batch_size, max_len, self.n_head, -1).transpose(1, 2) 
        k = k.view(batch_size, max_len, self.n_head, -1).transpose(1, 2) 
        v = v.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)  # b,h,t,c

        if self.add_rel_pos:
            AC = torch.matmul((q + self.r_r_bias), k.transpose(2, 3))  # b,h,t,t 
            BD = torch.matmul((self.r_w_bias + q), self.rel_pos_embedding)  # b,h,t,2t 
            BD = self._shift(BD)  # b,h,t,t 
            E  = torch.matmul(k, self.rel_pos_embedding)
            E  = self._shift(E).transpose(2, 3)
            attn_score = AC + BD + E
        else:
            attn_score = torch.matmul(q, k.transpose(2, 3))

        if self.add_scale:
            attn_score = attn_score / math.sqrt(self.head_dim)
        if skip_op and prev is not None:
            attn_score += prev  # connect previous attention 

        prev = attn_score 
        attn = torch.softmax(prev, dim=-1)
        attn = self.dropout(attn)
        v = torch.matmul(attn, v).transpose(1, 2).reshape(batch_size, max_len, d_model)  # b x n x l x d
        return v, prev

    def _shift(self, BD): 
        bsz, n_head, max_len, _ = BD.size()  # b,h,t,2t
        BD = F.pad(BD, (0, 1), "constant", 0)
        BD = BD.view(bsz, n_head, -1, max_len)
        BD = BD[:, :, :-1].view(bsz, n_head, max_len, -1)
        BD = BD[:, :, :, self.seq_len:]
        return BD 


class GLU_FeedForward(nn.Module):

    def __init__(self, d_model, kernel_size=5, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model*2, kernel_size=kernel_size, padding=kernel_size // 2)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = x.transpose(1, 2)  # b,t,c to b,c,t
        xx = self.dropout(self.conv(x))
        x, gate = torch.chunk(xx, chunks=2, dim=1)
        return (x * torch.sigmoid(gate)).transpose(1, 2)  # return:  b,t,c


class MLP_FeedForward(nn.Module):
    
    def __init__(self, d_model, feedforward_dim=2048, dropout=0.1):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Conv1d(d_model, feedforward_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(feedforward_dim, d_model, kernel_size=1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ffn(x.transpose(1, 2)).transpose(1, 2)


class CompactTransformer(nn.Module):

    def __init__(self, cfg, n_hidden, attn_op, skip_op, ffn_op):
        super().__init__()
        n_headers = cfg.transformer.n_headers
        p_dropout = cfg.transformer.p_dropout 
        seq_len = 32
        self.attn_op = attn_op 
        self.skip_op = skip_op
        self.ffn_op = ffn_op 

        if attn_op == 0:
            self.attn = RelativeMultiHeadAttnPro(n_hidden, n_headers, p_dropout, seq_len, add_scale=True,  add_rel_pos=True)
        elif attn_op == 1:
            self.attn = RelativeMultiHeadAttnPro(n_hidden, n_headers, p_dropout, seq_len, add_scale=True,  add_rel_pos=False)
        elif attn_op == 2:
            self.attn = RelativeMultiHeadAttnPro(n_hidden, n_headers, p_dropout, seq_len, add_scale=False, add_rel_pos=True)
        elif attn_op == 3:
            self.attn = RelativeMultiHeadAttnPro(n_hidden, n_headers, p_dropout, seq_len, add_scale=False, add_rel_pos=False)
        else:
            raise NotImplementedError 
        self.layer_norm1 = nn.LayerNorm(n_hidden)

        if ffn_op == 0:
            self.ffn = GLU_FeedForward(n_hidden, kernel_size=3, dropout=p_dropout)
        elif ffn_op == 1:
            self.ffn = MLP_FeedForward(n_hidden, feedforward_dim=int(n_hidden * 2), dropout=p_dropout)
        else:
            raise NotImplementedError 
        self.layer_norm2 = nn.LayerNorm(n_hidden)

    def forward(self, x, hidden):
        rmha, hidden = self.attn(x, hidden, self.skip_op)
        x = self.layer_norm1(x + rmha)
        x = self.layer_norm2(x + self.ffn(x))
        return x, hidden 


def encoder_layer(in_channels, out_channels, kernel_size=3, stride=2, padding=1):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(True)
    )


def decoder_layer(in_channels, out_channels, kernel_size=3, stride=1, padding=1, mode='nearest', scale_factor=None, size=None):
    align_corners = None if mode=='nearest' else True
    return nn.Sequential(
        nn.Upsample(size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners),
        nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(True)
    )


class CompactNet(nn.Module):

    TYPE_OPS = [CompactMBV3Block_k5e6, CompactMBV3Block_k3e6, CompactMBV3Block_k5e1, CompactMBV3Block_k3e1]

    def __init__(self, cfg, long_path, long_ops, attn_ops, skip_ops, ffn_ops):
        super().__init__()
        self.long_path = long_path
        self.long_ops = long_ops 
        self.attn_ops = attn_ops 
        self.skip_ops = skip_ops 
        self.ffn_ops  = ffn_ops 
        self.inplanes = 32
        self.n_hidden = 512
        self.n_attn = 512 
        self.n_k = 64
        self.max_length = cfg.max_length 

        self.stem = nn.Sequential(
            nn.Conv2d(3, self.inplanes, 3, 1, padding=1, bias=False),  # stride=1 for STR
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=True)
        )

        old_resolution = [32, 600]
        self.blocks = nn.ModuleList()
        for path, op in zip(long_path, long_ops):
            layer = self.TYPE_OPS[op](path, old_resolution)
            self.blocks.append(layer)
            old_resolution = path

        self.transformers = nn.ModuleList()
        for attn_op, skip_op, ffn_op in zip(attn_ops, skip_ops, ffn_ops):
            self.transformers.append(CompactTransformer(cfg, self.n_hidden, attn_op, skip_op, ffn_op))

        self.proj_q = nn.Linear(self.n_hidden, self.n_attn)
        self.proj_v = nn.Linear(self.n_hidden, self.n_attn)
        self.k_encoder = nn.Sequential(
            encoder_layer(self.n_hidden, self.n_k, stride=2),
            encoder_layer(self.n_k, self.n_k, stride=2),
            encoder_layer(self.n_k, self.n_k, stride=2),
            encoder_layer(self.n_k, self.n_k, stride=2),
        )
        self.k_decoder = nn.Sequential(
            decoder_layer(self.n_k, self.n_k, scale_factor=2, mode='nearest'),
            decoder_layer(self.n_k, self.n_k, scale_factor=2, mode='nearest'),
            decoder_layer(self.n_k, self.n_k, scale_factor=2, mode='nearest'),
            decoder_layer(self.n_k, self.n_attn, scale_factor=2, mode='nearest'),
        )
        self.pos_encoder = PositionalEncoding(self.n_hidden, dropout=0, max_len=self.max_length)
        self.classifier = nn.Linear(self.n_attn, cfg.n_classes)

    def forward(self, x):
        # feature extractor 
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = x.squeeze(dim=2).transpose(1, 2).contiguous()  # [batch_size, time_steps, n_hidden]
        hidden = None 
        for layer in self.transformers:
            x, hidden = layer(x, hidden)

        # key 
        k = x.transpose(1, 2)  # [batch_size, n_hidden, time_steps]
        features = []
        for i in range(0, len(self.k_encoder)):
            k = self.k_encoder[i](k)
            features.append(k)
        for i in range(0, len(self.k_decoder) - 1):
            k = self.k_decoder[i](k)
            k = k + features[len(self.k_decoder) - 2 - i]
        k = self.k_decoder[-1](k) # [batch_size, n_attn, time_steps]

        # query
        zeros = k.new_zeros((self.max_length, x.size(0), self.n_hidden))
        q = self.pos_encoder(zeros).permute(1, 0, 2)  # [batch_size, max_length, n_hidden]
        q = self.proj_q(q)                            # [batch_size, max_length, n_attn]

        # value 
        v = self.proj_v(x)  # [batch_size, time_steps, n_attn]

        # attention 
        attn_score = torch.bmm(q, k)  # [batch_size, max_length, time_steps]
        attn_score = torch.softmax(attn_score / (self.n_attn ** 0.5), dim=-1) 
        attn_vecs = torch.bmm(attn_score, v)  # [batch_size, max_length, n_attn]
        logits = self.classifier(attn_vecs)
        return logits 


class LabelMap_Attn:
    
    def __init__(self, cfg):
        self.cfg = cfg 
        self.voc = ['<eos>'] + list(string.digits + string.ascii_lowercase)
        self.char2id = dict(zip(self.voc, range(len(self.voc))))
        self.id2char = dict(zip(range(len(self.voc)), self.voc))
        self.n_classes = len(self.voc)
        self.eos_id = self.char2id['<eos>']

    def encode(self, text):
        assert isinstance(text, str)
        label = [self.char2id[c] for c in text]
        return label

    def decode(self, pred):
        if pred.ndim == 1:
            chars = []
            for index in pred:
                if index == self.eos_id:
                    break
                else:
                    chars.append(self.id2char[index])
            return "".join(chars)
        else:
            words = [self.decode(x) for x in pred]
            return words 


class LMDBDataset_Attn(torch.utils.data.Dataset):

    def __init__(self, label_map, root_dir, max_length=25, is_training=False, **kargs):
        super().__init__()
        with lmdb.open(root_dir) as env:
            txn = env.begin()
            self.n_samples = int(txn.get(b'num-samples'))
        self.root_dir = root_dir
        self.label_map = label_map
        self.img_h, self.img_w = 32, 128
        self.max_length = max_length 
        self.multiscales = False 
        self.is_training = is_training
        if self.is_training:
            self.augment_tfs = transforms.Compose([
                CVGeometry(degrees=45, translate=(0.0, 0.0), scale=(0.5, 2.), shear=(45, 15), distortion=0.5, p=0.5),
                CVDeterioration(var=20, degrees=6, factor=4, p=0.25),
                CVColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1, p=0.25)
            ])
        self.totensor = transforms.ToTensor()

    def open_lmdb(self):
        self.env = lmdb.open(self.root_dir)
        self.txn = self.env.begin()

    def __len__(self):
        return self.n_samples

    def aug_norm_img(self, img):
        img = cv2.resize(img, self.img_wh)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose((2, 0, 1))
        img = img.astype('float32')
        img = (img - 127.5) / 127.5
        return img 

    def resize_multiscales(self, img, borderType=cv2.BORDER_CONSTANT): 

        def _resize_ratio(img, ratio, fix_h=True):
            if ratio * self.img_w < self.img_h:
                if fix_h: 
                    trg_h = self.img_h
                else: 
                    trg_h = int(ratio * self.img_w)
                trg_w = self.img_w
            else: 
                trg_h, trg_w = self.img_h, int(self.img_h / ratio)
            img = cv2.resize(img, (trg_w, trg_h))
            pad_h, pad_w = (self.img_h - trg_h) / 2, (self.img_w - trg_w) / 2
            top, bottom = math.ceil(pad_h), math.floor(pad_h)
            left, right = math.ceil(pad_w), math.floor(pad_w)
            img = cv2.copyMakeBorder(img, top, bottom, left, right, borderType)
            return img
        
        if self.is_training: 
            if random.random() < 0.5:
                base, maxh, maxw = self.img_h, self.img_h, self.img_w
                h, w = random.randint(base, maxh), random.randint(base, maxw)
                return _resize_ratio(img, h/w)
            else: 
                return _resize_ratio(img, img.shape[0] / img.shape[1])  # keep aspect ratio
        else:  
            return _resize_ratio(img, img.shape[0] / img.shape[1])  # keep aspect ratio

    def resize(self, img):
        if self.multiscales: 
            return self.resize_multiscales(img, cv2.BORDER_REPLICATE)
        else: 
            return cv2.resize(img, (self.img_w, self.img_h))

    def __getitem__(self, index):
        if not hasattr(self, 'txn'):
            self.open_lmdb()

        try:
            img_key = b'image-%09d' % (index + 1)  # id starts from 1
            imgbuf = self.txn.get(img_key)
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning) # EXIF warning from TiffPlugin
                img = PIL.Image.open(buf).convert('RGB')
                if self.is_training:
                    img = self.augment_tfs(img)
                img = self.resize(np.array(img))
                img = self.totensor(img)
            img = img.sub_(0.5).div_(0.5)
        except:
            print('corrupted image for %s' % img_key)
            return self[(index + 1) % len(self)]

        label_key = b'label-%09d' % (index + 1)
        ori_label = ''.join(filter(
            lambda x: x in (string.digits + string.ascii_letters), 
            self.txn.get(label_key).decode()
        )).lower()
        label_list = self.label_map.encode(ori_label) + [self.label_map.eos_id]
        label_len = len(label_list)
        if label_len > self.max_length:
            return self[(index + 1) % len(self)]
        if label_len <= 0:
            print('word is none, skip %s' % img_key)
            return self[(index + 1) % len(self)]
        label = np.zeros((self.max_length, ), dtype=np.int)
        label[:label_len] = np.array(label_list)

        return img, label, label_len


def batch_collate(batch):
    images, labels, lengths = zip(*batch)
    b_lengths = torch.LongTensor(lengths)
    b_labels = torch.LongTensor(labels)
    b_images = torch.stack(images)
    return b_images, b_labels, b_lengths


def build_data(cfg):
    label_map = LabelMap_Attn(cfg)

    root_dir = cfg.DATASETS.ROOT_DIR
    
    train_folder = os.path.join(root_dir, 'ALL_REC_DATA')
    train_dataset  = LMDBDataset_Attn(label_map, train_folder, max_length=cfg.max_length, is_training=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
        batch_size=cfg.batch_size, num_workers=cfg.num_workers,
        shuffle=False, sampler=train_sampler, pin_memory=True,
        drop_last=True, collate_fn=batch_collate)

    test_data_folders = [os.path.join(root_dir, 'benchmark/test', name) for name in cfg.test_data_names]
    test_datasets = [LMDBDataset_Attn(label_map, data_folder, max_length=cfg.max_length, is_training=False) for data_folder in test_data_folders]

    def build_test_dataloader(dataset):
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_dataloader = torch.utils.data.DataLoader(dataset,
            batch_size=cfg.batch_size, num_workers=cfg.num_workers,
            shuffle=False, sampler=test_sampler, pin_memory=True,
            drop_last=False, collate_fn=batch_collate)
        return test_dataloader

    test_dataloaders = [build_test_dataloader(dataset) for dataset in test_datasets]
    return train_dataloader, test_dataloaders, label_map


class AttnLosses(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.loss_fn = torch.nn.CrossEntropyLoss().cuda(cfg.gpu)

    def _flatten(self, sources, lengths):
        return torch.cat([t[:l] for t, l in zip(sources, lengths)])

    def forward(self, pt_logits, gt_labels, gt_lengths):
        flat_gt_labels = self._flatten(gt_labels, gt_lengths)
        flat_pt_logits = self._flatten(pt_logits, gt_lengths)
        loss = self.loss_fn(flat_pt_logits, flat_gt_labels)
        return loss 


class TrainCompactController:

    def __init__(self, cfg, long_path, long_ops, attn_ops, skip_ops, ffn_ops):
        self.cfg = cfg 
        self.n_warmup_iters = cfg.n_warmup_iters

        if cfg.rank == 0:
            timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
            cfg.output_dir = os.path.join(cfg.output_dir, timestamp)
            cfg.ckpt_dir = os.path.join(cfg.output_dir, 'models')
            if not os.path.exists(cfg.ckpt_dir): os.makedirs(cfg.ckpt_dir)
            self.logger = get_logger(name='ParallelAttn-%d' % cfg.rank, log_file=os.path.join(cfg.output_dir, 'log.txt'))
            self.logger.info(cfg)
            self.logger.info(long_path)
            self.logger.info(long_ops)
            self.logger.info(attn_ops)
            self.logger.info(skip_ops)
            self.logger.info(ffn_ops)

        cfg.batch_size = int(cfg.batch_size / cfg.ngpus_per_node)

        torch.cuda.set_device(cfg.gpu)
        torch.backends.cudnn.benchmark = True
    
        # Init model 
        net = CompactNet(
            cfg, long_path, long_ops, attn_ops, skip_ops, ffn_ops)
        net.cuda(cfg.gpu)
        self.net = torch.nn.parallel.DistributedDataParallel(
            net, device_ids=[cfg.gpu])
        if cfg.rank == 0:
            self.logger.info(net)

        # Init training loss 
        self.loss_fn = AttnLosses(cfg)

        # Init dataset and dataloader 
        self.train_dataloader, self.test_dataloaders, self.label_map = build_data(cfg)
        if cfg.rank == 0:
            test_dataset_sizes = [len(dataloader.dataset) for dataloader in self.test_dataloaders]
            self.logger.info('train {}, test {}'.format(len(self.train_dataloader.dataset), test_dataset_sizes))

        # Init optimizer 
        self.optimizer = torch.optim.Adadelta(
            list(filter(lambda p: p.requires_grad, self.net.parameters())), 
            lr=cfg.init_lr, weight_decay=cfg.weight_decay)

        # Init learning rate scheduler 
        self.total_iters = len(self.train_dataloader) * self.cfg.total_epochs
        if self.n_warmup_iters > 0:
            warmup_with_cosine_lr = lambda epoch: \
                (epoch + 1) / self.n_warmup_iters if epoch < self.n_warmup_iters \
                else 0.5 * ( math.cos((epoch - self.n_warmup_iters) /(self.total_iters - self.n_warmup_iters) * math.pi) + 1)
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=warmup_with_cosine_lr)
        else:
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.total_iters, eta_min=0)

    def normalize_text(self, text):
        text = ''.join(filter(lambda x: x in (string.digits + string.ascii_letters), text)).lower()
        return text

    def test(self, dataloaders):
        results = {x: None for x in self.cfg.test_data_names}
        self.net.eval()
        with torch.no_grad():
            for dataloader, name in zip(dataloaders, self.cfg.test_data_names):
                check_list = []

                for imgs, labels, _ in dataloader:
                    imgs = imgs.cuda(self.cfg.gpu, non_blocking=True)
                    labels = labels.cuda(self.cfg.gpu, non_blocking=True)
                    output = self.net(imgs).argmax(-1)

                    output_list = [torch.zeros_like(output) for _ in range(self.cfg.world_size)]
                    dist.barrier()
                    dist.all_gather(output_list, output)
                    preds = torch.vstack(output_list).cpu().numpy()

                    labels_list = [torch.zeros_like(labels) for _ in range(self.cfg.world_size)]
                    dist.barrier()
                    dist.all_gather(labels_list, labels)
                    labels = torch.vstack(labels_list).cpu().numpy()

                    pred_strs = self.label_map.decode(preds)
                    label_strs = self.label_map.decode(labels)

                    if isinstance(label_strs, str):  # batch size 1
                        check_list.append(int(self.normalize_text(pred_strs) == self.normalize_text(label_strs)))
                    else:
                        for pred_str, label_str in zip(pred_strs, label_strs):
                            check_list.append(int(self.normalize_text(pred_str) == self.normalize_text(label_str)))

                # since the distribution sampler add extra samples to make it evenly divisible, remove it here
                n_total = min(len(check_list), len(dataloader.dataset))  
                acc = sum(check_list[:n_total]) / n_total
                results[name] = [acc, n_total]

        self.net.train()
        return results 

    def train(self):
        n_batches = len(self.train_dataloader)
        results = self.test(self.test_dataloaders)
        if self.cfg.rank == 0:
            self.logger.info('results: {}'.format(results))
            start_time = time.time()

        self.net.train()
        scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.cfg.total_epochs):

            self.train_dataloader.sampler.set_epoch(epoch)
            loss_counter = AverageMeter()
            self.net.train()

            for bid, (imgs, labels, label_lens) in enumerate(self.train_dataloader):
                self.optimizer.zero_grad()

                with torch.cuda.amp.autocast():
                    imgs = imgs.cuda(self.cfg.gpu, non_blocking=True)
                    labels = labels.cuda(self.cfg.gpu, non_blocking=True)
                    logits = self.net(imgs)
                    loss = self.loss_fn(logits, labels, label_lens)

                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                self.lr_scheduler.step()
                
                torch.distributed.barrier()
                reduced_loss = loss.clone()
                dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
                reduced_loss /= self.cfg.world_size
                loss_counter.update(reduced_loss.item(), imgs.size(0))

                if self.cfg.rank == 0 and (bid + 1) % self.cfg.print_every == 0:
                    lr = self.optimizer.param_groups[0]['lr']
                    self.logger.info('epoch [%d][%d/%d]\tLR %.5f\tLoss %.5f(%.5f)' % (
                        epoch, bid + 1, n_batches, lr, loss_counter.val, loss_counter.avg))

                n_iters = epoch * n_batches + bid
                if (n_iters + 1) % self.cfg.checkpoint_every == 0:
                    results = self.test(self.test_dataloaders)
                    if self.cfg.rank == 0:
                        torch.save(self.net.state_dict(), f=os.path.join(self.cfg.ckpt_dir, 'iter-%d.pth' % n_iters))
                        self.logger.info('elaspe {}s, results: {}'.format(time.time() - start_time, results))
            # end epoch
            
        # end training 
        results = self.test(self.test_dataloaders)
        if self.cfg.rank == 0:
            torch.save(self.net.state_dict(), f=os.path.join(self.cfg.ckpt_dir, 'final.pth'))
            self.logger.info('results: {}'.format(results))


def parser_args():
    # put all config here 
    cfg = CN()
    cfg.test_data_names = [
        'clovaai_test/%s' % x 
        for x in ['IIIT5k_3000', 'SVT', 'IC03_860', 
                  'IC03_867', 'IC13_857', 'IC13_1015', 
                  'IC15_1811', 'IC15_2077', 'SVTP', 'CUTE80']
    ]
    cfg.DATASETS = CN()
    cfg.DATASETS.ROOT_DIR = 'data'
    cfg.DATASETS.INPUT_SHAPE = [3, 32, 128]
    cfg.max_length = 25
    cfg.batch_size = 540
    cfg.num_workers = 4
    cfg.total_epochs = 12
    cfg.output_dir = '../work_dirs/Scene/main_dist_parallel_attention'
    cfg.init_lr = 1.0
    cfg.weight_decay = 1e-5
    cfg.n_warmup_iters = 1000
    cfg.n_classes = 37    # 0 = <eos> 1:37 = string.ascii_leters_lowcase + string.digits
    cfg.print_every = 100
    cfg.checkpoint_every = 1000 

    # transformer hyper-params
    cfg.transformer = CN()
    cfg.transformer.n_headers = 8
    cfg.transformer.p_dropout = 0.2

    # distributed training 
    cfg.seed = 1996
    cfg.distributed = True
    cfg.multiprocessing_distributed = True
    cfg.rank = 0
    cfg.dist_backend = 'nccl'
    cfg.dist_url = 'tcp://localhost:10088'
    return cfg


def main_worker(gpu, ngpus_per_node, cfg):
    cfg.gpu = gpu 

    if cfg.seed is not None:
        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

    if cfg.gpu is not None:
        print('use GPU: {} for training'.format(cfg.gpu))

    if cfg.gpu != 0:
        def print_pass(*args):
            pass 
        builtins.print = print_pass 

    cfg.rank = cfg.rank * ngpus_per_node + gpu 
    cfg.num_workers = int((cfg.num_workers + ngpus_per_node - 1) / ngpus_per_node)
    dist.init_process_group(backend=cfg.dist_backend, init_method=cfg.dist_url, world_size=cfg.world_size, rank=cfg.rank)

    long_path = [
        [32, 600], 
        [16, 600], [16, 600], [16, 600], [16, 600], [16, 600], [16, 600], [16, 600], [16, 600], [16, 600], 
        [8, 300], [8, 300], 
        [4, 300], [4, 300], [4, 300], [4, 300], [4, 300], 
        [2, 300], 
        [1, 150], [1, 150]
    ]
    long_ops = [
        0, 
        1, 0, 0, 1, 3, 0, 2, 2, 0, 
        1, 1, 
        1, 0, 0, 1, 1,
        0, 
        0, 0
    ]
    attn_ops = [0, 0, 1, 0]
    skip_ops = [0, 0, 1, 1]
    ffn_ops  = [1, 1, 1, 0]
    controller = TrainCompactController(cfg, long_path, long_ops, attn_ops, skip_ops, ffn_ops)
    controller.train()


if __name__ == '__main__':
    cfg = parser_args()
    cfg.ngpus_per_node = torch.cuda.device_count()
    cfg.world_size = cfg.ngpus_per_node
    mp.spawn(main_worker, nprocs=cfg.ngpus_per_node, args=(cfg.ngpus_per_node, cfg))
