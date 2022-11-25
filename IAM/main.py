import sys

sys.path.append('..')
import math
import os
import random
import time
from collections import OrderedDict

import editdistance as ed
import torch
import torch.nn as nn
import torch.nn.functional as F
from colorama import Fore
from datasets import Collate, LabelMap, NewIAMDataset
from utils import AverageMeter, get_logger
from utils.latency import compute_latency_ms_tensorrt, export_onnx
from warpctc_pytorch import CTCLoss
from yacs.config import CfgNode as CN

h_map_to_channels = {32: 16, 16: 16, 8: 32, 4: 64, 2:128, 1: 256}


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


def swish(x, inplace=True):
    return x.mul_(x.sigmoid()) if inplace else x.mul(x.sigmoid())


class MixDropout(nn.Module):
    def __init__(self, dropout_proba=0.4, dropout2d_proba=0.2):
        # Ref https://github.com/FactoDeepLearning/VerticalAttentionOCR/blob/master/basic/models.py
        super(MixDropout, self).__init__()
        self.dropout = nn.Dropout(dropout_proba)
        self.dropout2d = nn.Dropout2d(dropout2d_proba)

    def forward(self, x):
        if random.random() < 0.5:
            return self.dropout(x)
        return self.dropout2d(x)

    
class Swish(nn.Module):

    def __init__(self, inplace=True):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return x.mul_(x.sigmoid()) if self.inplace else x.mul(x.sigmoid())


class CompactMBV3Block_k5e6(nn.Module):
    kernel_size = 5
    ratio = 6

    '''expand + depthwise + pointwise'''
    def __init__(self, resolution, old_resolution, dropout=0.4):
        super().__init__()
        h, w = resolution 
        out_size = h_map_to_channels[h]
        expand_size = out_size * self.ratio 

        old_h, old_w = old_resolution
        old_c = h_map_to_channels[old_h]
        stride = [old_h // h, old_w // w]

        if stride != [1, 1]:
            self.fix_resolution = nn.Sequential(
                nn.Conv2d(old_c, out_size, 1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
                nn.ReLU(inplace=True)
            )
        else:
            self.fix_resolution = nn.Identity()

        self.conv1 = nn.Conv2d(out_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=self.kernel_size, 
                               stride=1, padding=self.kernel_size//2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.se = SeModule(expand_size, expand_size)
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)
        self.dropout = MixDropout(dropout_proba=dropout, dropout2d_proba=dropout / 2)

    def forward(self, x):
        pos = random.randint(1, 3)
        x = self.fix_resolution(x)
        identity = x
        out = swish(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = F.relu6(out)
        if pos == 1:
            out = self.dropout(out)
        out = self.bn3(self.conv3(out))
        out += identity 
        if pos == 2:
            out = self.dropout(out)
        return out


class CompactMBV3Block_k5e3(CompactMBV3Block_k5e6):
    kernel_size = 5
    ratio = 3


class CompactMBV3Block_k3e6(CompactMBV3Block_k5e6):
    kernel_size = 3
    ratio = 6


class CompactMBV3Block_k3e3(CompactMBV3Block_k5e6):
    kernel_size = 3
    ratio = 3


class CompactMBV3Block_k3e1(CompactMBV3Block_k5e6):
    kernel_size = 3
    ratio = 1


class CompactMBV3Block_k5e1(CompactMBV3Block_k5e6):
    kernel_size = 5
    ratio = 1


class CompactClassifierHead(nn.Module):
    
    def __init__(self, n_inc=1024, n_classes=81):
        super().__init__()
        self.classifier = nn.Linear(n_inc, n_classes)

    def forward(self, x):
        x = self.classifier(x)
        return x

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
        seq_len = 150
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


class CompactNet(nn.Module):

    TYPE_OPS = [CompactMBV3Block_k5e6, CompactMBV3Block_k5e1, CompactMBV3Block_k3e1]

    def __init__(self, cfg, long_path, long_ops, attn_ops, skip_ops, ffn_ops):
        super().__init__()
        self.inplanes = 16
        self.long_path = long_path
        self.long_ops = long_ops 
        self.attn_ops = attn_ops 
        self.skip_ops = skip_ops 
        self.ffn_ops  = ffn_ops 
        self.n_hidden = 256

        self.stem = nn.Sequential(
            nn.Conv2d(cfg.DATASETS.INPUT_SHAPE[0], self.inplanes, 3, 2, padding=1, bias=False),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=True)
        )

        old_resolution = [32, 600]
        self.blocks = nn.ModuleList()
        for path, op in zip(long_path, long_ops):
            layer = self.TYPE_OPS[op](path, old_resolution, dropout=cfg.p_mix_dropout)
            self.blocks.append(layer)
            old_resolution = path

        self.transformers = nn.ModuleList()
        for attn_op, skip_op, ffn_op in zip(attn_ops, skip_ops, ffn_ops):
            self.transformers.append(CompactTransformer(cfg, self.n_hidden, attn_op, skip_op, ffn_op))
        self.head = CompactClassifierHead(n_inc=self.n_hidden, n_classes=cfg.n_classes)

    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = x.squeeze(dim=2).transpose(1, 2).contiguous()
        hidden = None 
        for layer in self.transformers:
            x, hidden = layer(x, hidden)
        x = self.head(x)
        return x

    def inherit_state_dict(self, super_state_dict):
        state_dict_compact = OrderedDict()
        for k in self.state_dict().keys():
            if 'stem.' in k:
                key_super = k
            elif 'blocks.' in k:
                id_layer = int(k.split('.')[1])
                cur_r = self.long_path[id_layer]
                op = self.long_ops[id_layer]
                if 'fix_resolution' in k:
                    pre_r = [32, 600]  if id_layer == 0 else self.long_path[id_layer - 1]
                    key_super = 'blocks.%d.layers.%d.%d_%d.mix_op.%d.fix_resolutions.%d_%d.%s' % (
                        id_layer // 5, id_layer % 5, cur_r[0], cur_r[1], op, pre_r[0], pre_r[1],
                        k.split('.', 3)[-1])
                else:
                    key_super = 'blocks.%d.layers.%d.%d_%d.mix_op.%d.%s' % (
                        id_layer // 5, id_layer % 5, cur_r[0], cur_r[1], op,
                        k.split('.', 2)[-1])
            elif 'transformers.' in k:
                id_layer = int(k.split('.')[1])
                if 'attn' in k or 'layer_norm1' in k:
                    key_super = 'blocks.4.layers.%d.%s.%d.%s' % (
                        id_layer, 'attn_blocks' if 'attn' in k else 'layer_norm1',
                        self.attn_ops[id_layer], k.split('.', 3)[-1])
                elif 'ffn' in k or 'layer_norm2' in k:
                    key_super = 'blocks.4.layers.%d.%s.%d.%s' % (
                        id_layer, 'ffn_blocks' if 'ffn' in k else 'layer_norm2',
                        self.ffn_ops[id_layer], k.split('.', 3)[-1])
                else:
                    raise ValueError 
            elif 'head.' in k:
                key_super = k.replace('head.', 'heads.4.')
            state_dict_compact[k] = super_state_dict[key_super]
        # loading inherited weight 
        self.load_state_dict(state_dict_compact)


def build_data(cfg, logger=None):
    root_dir = cfg.DATASETS.ROOT_DIR
    label_map = LabelMap(character_set_file=os.path.join(
        root_dir, 'syms.txt'))
    train_dataset = NewIAMDataset(label_map,
        root_dir=root_dir,
        mode='train',
        input_shape=cfg.DATASETS.INPUT_SHAPE,
        p_aug=cfg.DATASETS.INIT_P_AUG, 
        # text_img_distort=cfg.DATASETS.text_img_distort
        )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=cfg.DATASETS.BATCH_SIZE, 
        shuffle=True, 
        num_workers=16, 
        collate_fn=Collate())

    val_dataset = NewIAMDataset(label_map,
        root_dir=root_dir,
        mode='valid',
        input_shape=cfg.DATASETS.INPUT_SHAPE)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=cfg.DATASETS.BATCH_SIZE * 2,
        shuffle=False, 
        num_workers=16, 
        collate_fn=Collate())

    test_dataset = NewIAMDataset(label_map,
        root_dir=root_dir,
        mode='test',
        input_shape=cfg.DATASETS.INPUT_SHAPE)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=cfg.DATASETS.BATCH_SIZE * 2, 
        shuffle=False, 
        num_workers=16, 
        collate_fn=Collate())
    if logger:
        logger.info('n_train: %d, n_val: %d, n_test %d' % (
            len(train_dataset), len(val_dataset), len(test_dataset)))
    return train_dataloader, val_dataloader, test_dataloader, label_map


from default_cfg import cfg as d_cfg
from models import build_recognition_model

d_cfg.PRINT_EVERY = 100 
d_cfg.SEED = 10 
d_cfg.merge_from_file('../config_yamls/0330/baseline_woAug.yaml')


class TrainCompactController:

    def __init__(self, cfg, long_path, long_ops, attn_ops, skip_ops, ffn_ops):
        self.cfg = cfg 
        self.warmup_epochs = cfg.n_warmup_epochs
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        cfg.output_dir = os.path.join(cfg.output_dir, timestamp)
        cfg.ckpt_dir = os.path.join(cfg.output_dir, 'models')
        if not os.path.exists(cfg.ckpt_dir): os.makedirs(cfg.ckpt_dir)
        self.logger = get_logger(name='BeatSOTA', log_file=os.path.join(cfg.output_dir, 'log.txt'))
        self.logger.info(cfg)
        self.logger.info(long_path)
        self.logger.info(long_ops)
        self.logger.info(attn_ops)
        self.logger.info(skip_ops)
        self.logger.info(ffn_ops)

        torch.backends.cudnn.benchmark = True

        # build net/optimizer/lr_scheduler/loss_fn/dataloader etc 
        self.net = CompactNet(cfg, long_path, long_ops, attn_ops, skip_ops, ffn_ops)
        self.logger.info(self.net)
        self.net = self.net.cuda()
        self.net = nn.DataParallel(self.net)
        self.optimizer = torch.optim.Adadelta(self.net.parameters(), 
            lr=cfg.init_lr, weight_decay=cfg.weight_decay)
        if self.warmup_epochs > 0:
            warmup_with_cosine_lr = lambda epoch: \
                (epoch + 1) / self.warmup_epochs if epoch < self.warmup_epochs \
                else 0.5 * ( math.cos((epoch - self.warmup_epochs) /(cfg.total_epochs - self.warmup_epochs) * math.pi) + 1)
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=warmup_with_cosine_lr)
        else:
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=cfg.total_epochs, eta_min=0)
        self.loss_fn = CTCLoss(size_average=True).cuda()
        self.train_dataloader, self.val_dataloader, self.test_dataloader, self.label_map = build_data(cfg, self.logger)

    def test(self, dataloader):
        self.net.eval()
        with torch.no_grad():
            total_c = distance_c = 0
            total_w = distance_w = 0
            for batch in dataloader:
                imgs, labels, label_lens = batch[:3]
                output = self.net(imgs.cuda())
                pred   = output.argmax(-1).cpu().numpy()  # b, t 
                pred_texts  = self.label_map.decode(pred, raw=False) 
                label_texts = self.label_map.decode_label(labels, label_lens) 
                for i in range(0, len(pred_texts)):
                    # CER 
                    pred_text   = pred_texts[i].strip()
                    label_text  = label_texts[i].strip()
                    distance_c += ed.eval(label_text, pred_text)
                    total_c    += len(label_text)
                    # WER 
                    all_words = []
                    label_words = label_text.split(' ')
                    pred_words = pred_text.split(' ')
                    for w in label_words + pred_words:
                        if w not in all_words:
                            all_words.append(w)
                    l_words = [all_words.index(x) for x in label_words]
                    p_words = [all_words.index(x) for x in pred_words]
                    distance_w += ed.eval(l_words, p_words)
                    total_w    += len(label_words)
        CER = distance_c / total_c 
        WER = distance_w / total_w
        return CER, WER 

    def train(self):
        test_cers = []
        self.net.train()
        
        n_batches = len(self.train_dataloader)
        for epoch in range(self.cfg.total_epochs):
            if (epoch + 1) > self.warmup_epochs:
                self.train_dataloader.dataset.p_aug = cfg.DATASETS.INIT_P_AUG * (
                    self.optimizer.param_groups[0]['lr'] / cfg.init_lr)
            else:
                self.train_dataloader.dataset.p_aug = 0

            # train one epoch 
            self.net.train()
            loss_counter = AverageMeter()
            n_batches = len(self.train_dataloader)
            for bid, (imgs, labels, label_lens) in enumerate(self.train_dataloader):
                imgs = imgs.cuda()
                output = self.net(imgs)
                probs = output.transpose(0, 1).contiguous().cuda()
                label_size = label_lens
                probs_size = torch.IntTensor([probs.size(0)] * probs.size(1))
                probs.requires_grad_(True)
                loss = self.loss_fn(probs, labels, probs_size, label_size)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 5, 2)
                self.optimizer.step()

                loss_counter.update(loss.item(), imgs.size(0))
                if bid % self.cfg.print_every == 0:
                    lr = self.optimizer.param_groups[0]['lr']
                    self.logger.info('Epoch %d (%d / %d), lr %.5f, p_aug %.5f, loss %.5f' % (
                        epoch, bid, n_batches, lr, self.train_dataloader.dataset.p_aug, loss_counter.avg))
            # end of train one epoch
            self.lr_scheduler.step()

            train_cer, train_wer = self.test(self.train_dataloader)
            test_cer,  test_wer  = self.test(self.test_dataloader)
            val_cer,   val_wer   = self.test(self.val_dataloader)
            self.logger.info(Fore.YELLOW + 
                '%d/%d: train (CER %.5f / WER %.5f), val (CER %.5f / WER %.5f), test (CER %.5f / WER %.5f)' % (
                epoch, cfg.total_epochs, train_cer, train_wer, val_cer, val_wer, test_cer, test_wer) + Fore.WHITE) 
            test_cers.append(test_cer)
        # end of 
        self.logger.info('Report: minCER %.5f, meanCER %.5f' % (min(test_cers), sum(test_cers[-10:]) / 10))
        torch.save(self.net.state_dict(), os.path.join(self.cfg.ckpt_dir, 'final.pth'))


def parser_args():
    # put all config here 
    cfg = CN()
    cfg.DATASETS = CN()
    cfg.DATASETS.ROOT_DIR = '../new_data'
    cfg.DATASETS.INPUT_SHAPE = [1, 64, 1200]
    cfg.DATASETS.BATCH_SIZE = 64
    cfg.DATASETS.INIT_P_AUG = 0.5
    cfg.DATASETS.text_img_distort = True
    cfg.total_epochs = 1000
    cfg.output_dir = '../work_dirs/train_compact/fixfix'
    cfg.init_lr = 0.8
    cfg.weight_decay = 1e-5
    cfg.n_warmup_epochs = 10
    cfg.n_classes = 81  # 0 = blank 1:80 = class 
    cfg.print_every = 100
    # transformer
    cfg.transformer = CN()
    cfg.transformer.n_headers = 8
    cfg.transformer.p_dropout = 0.2
    # cnn
    cfg.p_mix_dropout = 0.1
    return cfg


if __name__ == '__main__':
    long_path = [[32, 600], [32, 600], [16, 600], [16, 600], [16, 600], [16, 600], [16, 600], [16, 600], [8, 600], [8, 600], [8, 600], [8, 600], [4, 600], [2, 300], [2, 300], [2, 300], [2, 300], [2, 300], [1, 150], [1, 150]]
    long_ops = [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0]
    attn_ops = [1, 1, 2, 0]
    skip_ops = [0, 1, 0, 1]
    ffn_ops = [0, 1, 0, 0]
    cfg = parser_args()
    # controller = TrainCompactController(cfg, long_path, long_ops, attn_ops, skip_ops, ffn_ops)
    # controller.train()

    # Testing latency 
    net = CompactNet(cfg, long_path, long_ops, attn_ops, skip_ops, ffn_ops)
    net = net.cuda()
    input_size = [1, 1, 64, 1200]
    onnx_file = export_onnx(net, input_size=input_size, onnx_file='IAM.onnx')
    latency = compute_latency_ms_tensorrt(onnx_file, input_size=input_size, num_iters=1000, verbose=False, mode='FP32')
    print(latency)
