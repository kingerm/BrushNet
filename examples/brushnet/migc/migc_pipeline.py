import glob
import random
import time
from typing import Any, Callable, Dict, List, Optional, Union
# import moxing as mox
import numpy as np
import torch
from diffusers.loaders import TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention_processor import Attention
from diffusers.pipelines.stable_diffusion import (
    StableDiffusionPipeline,
    StableDiffusionPipelineOutput,
    StableDiffusionSafetyChecker,
)
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import logging
from PIL import Image, ImageDraw, ImageFont
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
import inspect
import os
import math
import torch.nn as nn
import torch.nn.functional as F
# from utils import load_utils
import argparse
import yaml
import cv2
import math
from migc.migc_arch import MIGC, NaiveFuser
from scipy.ndimage import uniform_filter, gaussian_filter
from einops import rearrange
from torch_kmeans import KMeans
import nltk

logger = logging.get_logger(__name__)

class AttentionStore:  # 要有一个人来继承他才行  # 所以要把self.mean_map啥的放到这下面！
    @staticmethod
    def get_empty_store():
        return {"down": [], "mid": [], "up": []}

    EPSILON = 1e-5
    FILTER_TAGS = {
        'CC', 'CD', 'DT', 'EX', 'IN', 'LS', 'MD', 'PDT', 'POS', 'PRP', 'PRP$', 'RP', 'TO', 'UH', 'WDT', 'WP', 'WRB'}
    TAG_RULES = {'left': 'IN', 'right': 'IN', 'top': 'IN', 'bottom': 'IN'}
    def __init__(self, attn_res=[64 * 64, 32 * 32, 16 * 16, 8 * 8],
                 cross_mask_layers=None,
                 self_mask_layers=None,
                 eos_token_index=None,
                 filter_token_indices=None,
                 leading_token_indices=None,
                 mask_cross_during_guidance=True,
                 mask_eos=True,
                 cross_loss_coef=1.5,
                 self_loss_coef=0.5,
                 max_guidance_iter=15,
                 max_guidance_iter_per_step=5,
                 start_step_size=18,
                 end_step_size=5,
                 loss_stopping_value=0.2,
                 min_clustering_step=15,
                 cross_mask_threshold=0.2,
                 self_mask_threshold=0.2,
                 delta_refine_mask_steps=5,  # 5
                 pca_rank=None,
                 num_clusters=None,
                 num_clusters_per_box=3,
                 max_resolution=None,
                 map_dir=None,
                 debug=False,
                 delta_debug_attention_steps=20,
                 delta_debug_mask_steps=5,
                 debug_layers=None,
                 saved_resolution=64,
                 ):
        """
        Initialize an empty AttentionStore :param step_index: used to visualize only a specific step in the diffusion
        process
        """
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.cur_step = 0
        self.attn_res = attn_res
        # 下面是ba的参数部分，只有放在这里才能记录吧
        self.prompts = ['']
        self.subject_token_indices = [[2, 3], [6, 7]]  # 这个不想再写到外面去了，就放在里面吧
        self.filter_token_indices = None  # 初始化就是none
        self.cross_mask_layers = {14, 15, 16, 17, 18, 19}
        self.self_mask_layers = {14, 15, 16, 17, 18, 19}
        self.eos_token_index = eos_token_index
        self.filter_token_indices = filter_token_indices
        self.leading_token_indices = [3, 7]#leading_token_indices
        self.mask_cross_during_guidance = mask_cross_during_guidance
        self.mask_eos = mask_eos
        self.cross_loss_coef = cross_loss_coef
        self.self_loss_coef = self_loss_coef
        self.max_guidance_iter = max_guidance_iter
        self.max_guidance_iter_per_step = max_guidance_iter_per_step
        self.start_step_size = start_step_size
        self.step_size_coef = (end_step_size - start_step_size) / max_guidance_iter
        self.loss_stopping_value = loss_stopping_value
        self.min_clustering_step = min_clustering_step
        self.cross_mask_threshold = cross_mask_threshold
        self.self_mask_threshold = self_mask_threshold

        self.delta_refine_mask_steps = delta_refine_mask_steps
        self.pca_rank = pca_rank
        num_clusters = 2 * num_clusters_per_box if num_clusters is None else num_clusters  # 这个2的位置本来是len(boxs)
        self.clustering = KMeans(n_clusters=num_clusters, num_init=100)
        self.centers = None
        self.saved_resolution = saved_resolution

        self.optimized = False
        self.cross_foreground_values = []
        self.self_foreground_values = []
        self.cross_background_values = []
        self.self_background_values = []
        self.mean_cross_map = 0
        self.num_cross_maps = 0
        self.mean_self_map = 0
        self.num_self_maps = 0
        self.self_masks = None

    def __call__(self, attn, is_cross: bool, place_in_unet: str, q, k, v, bboxes, num_heads, scale, BaSteps, hidden_states):
        if is_cross:  # 这个主要是储存attention_probs，并不会改变变量值，所以可以调转顺序
            if attn.shape[1] in self.attn_res:
                self.step_store[place_in_unet].append(attn)

        # 在下面把ba那一套给接上就行了，十分的easy
        if self.cur_step >= BaSteps:
            batch_size = q.size(0) // num_heads  # 用的是forward而不是__call__!
            n = q.size(1)
            d = k.size(1)
            dtype = q.dtype
            device = q.device
            if is_cross:
                # return hidden_states
                masks = self._hide_other_subjects_from_tokens(batch_size // 2, n, d, bboxes, BaSteps, dtype, device)
            else:
                masks = self._hide_other_subjects_from_subjects(batch_size // 2, bboxes, BaSteps, n, dtype, device)
            # 这里的masks是attention的masks，所以size为(2, 1, 64, 64)。但migc并没有用到attention mask?得再确定一下
            sim = torch.einsum('b i d, b j d -> b i j', q, k) * scale  # 这是做scale-dot production
            sim = sim.reshape(batch_size, num_heads, n, d) + masks
            attn = sim.reshape(-1, n, d).softmax(-1)
            self._save(attn, is_cross, num_heads)
            out = torch.bmm(attn, v)
            hidden_states = rearrange(out, '(b h) n d -> b n (h d)', h=num_heads)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return hidden_states
    def between_steps(self):
        self.attention_store = self.step_store
        self.step_store = self.get_empty_store()

    def maps(self, block_type: str):
        return self.attention_store[block_type]

    def before_step(self):
        self.clear_values()
        if self.cur_step == 0:
            self._determine_tokens()
    def _determine_tokens(self):
        self._determine_eos_token()
        self._determine_filter_tokens()
        self._determine_leading_tokens()
    def _determine_eos_token(self):
        tokens = self._tokenize()
        eos_token_index = len(tokens) + 1
        if self.eos_token_index is None:
            self.eos_token_index = eos_token_index
        elif eos_token_index != self.eos_token_index:
            raise ValueError(f'Wrong EOS token index. Tokens are: {tokens}.')

    def _determine_filter_tokens(self):
        if self.filter_token_indices is not None:
            return

        tags = self._tag_tokens()
        self.filter_token_indices = [i + 1 for i, tag in enumerate(tags) if tag in type(self).FILTER_TAGS]

    def _determine_leading_tokens(self):
        if self.leading_token_indices is not None:
            return

        tags = self._tag_tokens()
        leading_token_indices = []
        for indices in self.subject_token_indices:
            subject_noun_indices = [i for i in indices if tags[i - 1].startswith('NN')]
            leading_token_candidates = subject_noun_indices if len(subject_noun_indices) > 0 else indices
            leading_token_indices.append(leading_token_candidates[-1])

        self.leading_token_indices = leading_token_indices
    def reset(self):
        self.cur_att_layer = 0
        self.step_store = self.get_empty_store()
        self.attention_store = {}
    def clear_values(self, include_maps=False):
        lists = (
            self.cross_foreground_values,
            self.self_foreground_values,
            self.cross_background_values,
            self.self_background_values,
        )

        for values in lists:
            values.clear()

        if include_maps:
            self.mean_cross_map = 0
            self.num_cross_maps = 0
            self.mean_self_map = 0
            self.num_self_maps = 0
    def _tag_tokens(self):
        tagged_tokens = nltk.pos_tag(self._tokenize())
        return [type(self).TAG_RULES.get(token, tag) for token, tag in tagged_tokens]
    def _tokenize(self):
        ids = self.model.tokenizer.encode(self.prompts[0])
        tokens = self.model.tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=True)
        return [token[:-4] for token in tokens]  # remove ending </w>
    def _save(self, attn, is_cross, num_heads):
        _, attn = attn.chunk(2)
        attn = attn.reshape(-1, num_heads, *attn.shape[-2:])  # b h n k

        self._save_mask_maps(attn, is_cross)

    def _save_mask_maps(self, attn, is_cross):
        if (
                (self.optimized) or
                (is_cross and self.cur_att_layer not in self.cross_mask_layers) or
                ((not is_cross) and (self.cur_att_layer not in self.self_mask_layers))
        ):
            return
        if is_cross:
            attn = attn[..., self.leading_token_indices]
            mean_map = self.mean_cross_map
            num_maps = self.num_cross_maps
        else:
            mean_map = self.mean_self_map
            num_maps = self.num_self_maps

        num_maps += 1
        attn = attn.mean(dim=1)  # mean over heads
        mean_map = ((num_maps - 1) / num_maps) * mean_map + (1 / num_maps) * attn
        if is_cross:
            self.mean_cross_map = mean_map
            self.num_cross_maps = num_maps
        else:
            self.mean_self_map = mean_map  # 这只有14 16 18层会记录
            self.num_self_maps = num_maps

    # 下面是mask refine部分相关函数
    def _hide_other_subjects_from_tokens(self, batch_size, n, d, bboxes, BaSteps, dtype, device):  # b h i j
        resolution = int(n ** 0.5)
        subject_masks, background_masks = self._obtain_masks(resolution, bboxes, BaSteps, batch_size=batch_size,
                                                             device=device)  # b s n
        include_background = self.optimized or (
                not self.mask_cross_during_guidance and self.cur_step < self.max_guidance_iter_per_step)
        subject_masks = torch.logical_or(subject_masks,
                                         background_masks.unsqueeze(1)) if include_background else subject_masks
        min_value = torch.finfo(dtype).min
        sim_masks = torch.zeros((batch_size, n, d), dtype=dtype, device=device)  # b i j
        for token_indices in (*self.subject_token_indices, self.filter_token_indices):
            sim_masks[:, :, token_indices] = min_value

        for batch_index in range(batch_size):
            for subject_mask, token_indices in zip(subject_masks[batch_index], self.subject_token_indices):
                for token_index in token_indices:
                    sim_masks[batch_index, subject_mask, token_index] = 0

        if self.mask_eos and not include_background:
            for batch_index, background_mask in zip(range(batch_size), background_masks):
                sim_masks[batch_index, background_mask, self.eos_token_index] = min_value

        return torch.cat((torch.zeros_like(sim_masks), sim_masks)).unsqueeze(1)

    def _hide_other_subjects_from_subjects(self, batch_size, bboxes, BaSteps, n, dtype, device):  # b h i j
        resolution = int(n ** 0.5)
        subject_masks, background_masks = self._obtain_masks(resolution, bboxes, BaSteps, batch_size=batch_size,
                                                             device=device)  # b s n
        min_value = torch.finfo(dtype).min
        sim_masks = torch.zeros((batch_size, n, n), dtype=dtype, device=device)  # b i j
        for batch_index, background_mask in zip(range(batch_size), background_masks):
            all_subject_mask = ~background_mask.unsqueeze(0) * ~background_mask.unsqueeze(1)
            sim_masks[batch_index, all_subject_mask] = min_value

        for batch_index in range(batch_size):
            for subject_mask in subject_masks[batch_index]:
                subject_sim_mask = sim_masks[batch_index, subject_mask]
                condition = torch.logical_or(subject_sim_mask == 0, subject_mask.unsqueeze(0))
                sim_masks[batch_index, subject_mask] = torch.where(condition, 0, min_value).to(dtype=dtype)

        return torch.cat((sim_masks, sim_masks)).unsqueeze(1)

    # obtain_masks有点像是refine mask的函数哦
    def _obtain_masks(self, resolution, bboxes, BaSteps, return_boxes=False, return_existing=False, batch_size=None,
                      device=None):
        return_boxes = return_boxes or (return_existing and self.self_masks is None)
        # 这里就是开始refine mask的步骤！当return_boxes = False且现在的步数大于最小聚类步数
        if return_boxes or self.cur_step <= BaSteps:
            masks = self._convert_boxes_to_masks(resolution, bboxes, device=device).unsqueeze(0)
            if batch_size is not None:
                masks = masks.expand(batch_size, *masks.shape[1:])
        else:  # 这里就是开始refine mask的步骤！当return_boxes = False且现在的步数大于最小聚类步数
            masks = self._obtain_self_masks(resolution, bboxes, BaSteps, return_existing=return_existing)  # 获得聚类后的mask后再返回
            if device is not None:
                masks = masks.to(device=device)

        background_mask = masks.sum(dim=1) == 0
        return masks, background_mask

    def _convert_boxes_to_masks(self, resolution, bboxes, device=None):  # s n
        boxes = torch.zeros(len(bboxes[0]), resolution, resolution, dtype=bool, device=device)
        for i, box in enumerate(bboxes[0]):
            x0, x1 = box[0] * resolution, box[2] * resolution
            y0, y1 = box[1] * resolution, box[3] * resolution

            boxes[i, round(y0): round(y1), round(x0): round(x1)] = True

        return boxes.flatten(start_dim=1)

    def _obtain_self_masks(self, resolution, bboxes, BaSteps, return_existing=False):
        if (
                (self.self_masks is None) or
                (
                        (self.cur_step % self.delta_refine_mask_steps == 0) and
                        (self.cur_att_layer == 0) and
                        (not return_existing)
                )
        ):
            self.self_masks = self._fix_zero_masks(self._build_self_masks(bboxes, BaSteps), bboxes)  # 这是每个ith都修整mask

        b, s, n = self.self_masks.shape
        mask_resolution = int(n ** 0.5)
        self_masks = self.self_masks.reshape(b, s, mask_resolution, mask_resolution).float()
        self_masks = F.interpolate(self_masks, resolution, mode='nearest-exact')
        return self_masks.flatten(start_dim=2).bool()

    # build_self_masks更像是生成sa mask的函数！
    def _build_self_masks(self, bboxes, BaSteps):
        c, clusters = self._cluster_self_maps()  # b n
        cluster_masks = torch.stack([(clusters == cluster_index) for cluster_index in range(c)], dim=2)  # b n c
        cluster_area = cluster_masks.sum(dim=1, keepdim=True)  # b 1 c

        n = clusters.size(1)
        resolution = int(n ** 0.5)
        cross_masks = self._obtain_cross_masks(resolution, bboxes, BaSteps)  # b s n
        cross_mask_area = cross_masks.sum(dim=2, keepdim=True)  # b s 1

        intersection = torch.bmm(cross_masks.float(), cluster_masks.float())  # b s c
        min_area = torch.minimum(cross_mask_area, cluster_area)  # b s c
        score_per_cluster, subject_per_cluster = torch.max(intersection / min_area, dim=1)  # b c
        subjects = torch.gather(subject_per_cluster, 1, clusters)  # b n
        scores = torch.gather(score_per_cluster, 1, clusters)  # b n

        s = cross_masks.size(1)
        self_masks = torch.stack([(subjects == subject_index) for subject_index in range(s)], dim=1)  # b s n
        scores = scores.unsqueeze(1).expand(-1, s, n)  # b s n
        self_masks[scores < self.self_mask_threshold] = False
        # self._save_maps(self_masks, 'self_masks')
        return self_masks

    def _cluster_self_maps(self):  # b s n
        self_maps = self._compute_maps(self.mean_self_map)  # b n m  # self.mean_self_map一开始是0，为int？这个正常吗？
        if self.pca_rank is not None:
            dtype = self_maps.dtype
            _, _, eigen_vectors = torch.pca_lowrank(self_maps.float(), self.pca_rank)
            self_maps = torch.matmul(self_maps, eigen_vectors.to(dtype=dtype))
        # 这就是把sa maps丢进去聚类的
        clustering_results = self.clustering(self_maps, centers=self.centers)
        self.clustering.num_init = 1  # clustering is deterministic after the first time
        self.centers = clustering_results.centers
        clusters = clustering_results.labels
        num_clusters = self.clustering.n_clusters
        # self._save_maps(clusters / num_clusters, f'clusters')
        return num_clusters, clusters

    def _obtain_cross_masks(self, resolution, bboxes, BaSteps, scale=10):
        maps = self._compute_maps(self.mean_cross_map, resolution=resolution)  # b n k
        maps = F.sigmoid(scale * (maps - self.cross_mask_threshold))
        maps = self._normalize_maps(maps, reduce_min=True)
        maps = maps.transpose(1, 2)  # b k n
        existing_masks, _ = self._obtain_masks(
            resolution, bboxes, BaSteps, return_existing=True, batch_size=maps.size(0), device=maps.device)
        maps = maps * existing_masks.to(dtype=maps.dtype)
        # self._save_maps(maps, 'cross_masks')
        return maps

    def _fix_zero_masks(self, masks, bboxes):
        b, s, n = masks.shape
        resolution = int(n ** 0.5)
        boxes = self._convert_boxes_to_masks(resolution, bboxes, device=masks.device)  # s n

        for i in range(b):
            for j in range(s):
                if masks[i, j].sum() == 0:
                    print('******Found a zero mask!******')
                    for k in range(s):
                        masks[i, k] = boxes[j] if (k == j) else masks[i, k].logical_and(~boxes[j])

        return masks

    def _compute_maps(self, maps, resolution=None):  # b n k
        if resolution is not None:
            b, n, k = maps.shape
            original_resolution = int(n ** 0.5)
            maps = maps.transpose(1, 2).reshape(b, k, original_resolution, original_resolution)
            maps = F.interpolate(maps, resolution, mode='bilinear', antialias=True)
            maps = maps.reshape(b, k, -1).transpose(1, 2)

        maps = self._normalize_maps(maps)
        return maps

    @classmethod
    def _normalize_maps(cls, maps, reduce_min=False):  # b n k
        max_values = maps.max(dim=1, keepdim=True)[0]
        min_values = maps.min(dim=1, keepdim=True)[0] if reduce_min else 0
        numerator = maps - min_values
        denominator = max_values - min_values + cls.EPSILON
        return numerator / denominator

def get_sup_mask(mask_list):
    or_mask = np.zeros_like(mask_list[0])
    for mask in mask_list:
        or_mask += mask
    or_mask[or_mask >= 1] = 1
    sup_mask = 1 - or_mask
    return sup_mask

# 这个migcprocessor并不能继承attentionstore。所以要用ba的话，得另外写一个函数继承attentionstore才行！
class MIGCProcessor(nn.Module):  # 这看起来就像是migc的net！得好好研究一下  # 这是嵌入在unet里面的attention processor
    def __init__(self, config, attnstore, place_in_unet, scale):
        super().__init__()
        self.attnstore = attnstore
        self.place_in_unet = place_in_unet
        self.not_use_migc = config['not_use_migc']
        self.naive_fuser = NaiveFuser()
        self.embedding = {}
        self.scale = scale
        if not self.not_use_migc:
            self.migc = MIGC(config['C'])  # 这里是进入migc_arch的MIGC类中进行初始化

    def __call__(
            self,
            attn: Attention,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            prompt_nums=[],
            bboxes=[],  # bboxes是啥？应该是xyxy或者其embedding
            ith=None,
            embeds_pooler=None,
            timestep=None,
            height=512,
            width=512,
            MIGCsteps=20,
            NaiveFuserSteps=-1,
            BaSteps=-1,
            ca_scale=None,
            ea_scale=None,
            sac_scale=None,
            use_sa_preserve=False,
            sa_preserve=False,
    ):
        batch_size, sequence_length, _ = hidden_states.shape  # sequence_length = H * W
        hidden_states_ba =hidden_states
        assert (batch_size == 2, "We currently only implement sampling with batch_size=1, \
               and we will implement sampling with batch_size=N as soon as possible.")
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )

        instance_num = len(bboxes[0])

        if ith >= MIGCsteps:  # migcsteps为25，当处于inference后半段时，不使用migc
            not_use_migc = True
        else:
            not_use_migc = self.not_use_migc
        is_vanilla_cross = (
                    not_use_migc and ith >= NaiveFuserSteps)  # 如果把这个拉满，那么就不会再有vanilla cross，相当于后半段inference全程作用naivefuser
        if ith > MIGCsteps:
            not_use_migc = True
        else:
            not_use_migc = self.not_use_migc
        if ith > MIGCsteps:
            not_use_migc = True
        else:
            not_use_migc = self.not_use_migc
        is_vanilla_cross = (not_use_migc and ith > NaiveFuserSteps)
        if instance_num == 0:
            is_vanilla_cross = True

        # 这代表着：如果是cross层，那么就把合起来的encoder_hidden_states给劈开
        if encoder_hidden_states is not None:
            encoder_hidden_states_ba = torch.concat([encoder_hidden_states[-1].unsqueeze(0), encoder_hidden_states[0].unsqueeze(0)])  # (2, 77, 768)
            encoder_hidden_states = encoder_hidden_states[:-1]  # (prompts_num+1, 77, 768)
        else:
            encoder_hidden_states_ba = hidden_states

        is_cross = encoder_hidden_states is not None
        # 存下SA的K V，目前的ori_hidden_state为(2, HW, 320)
        ori_hidden_states = hidden_states.clone()

        # Only Need Negative Prompt and Global Prompt.
        # "In other Cross-Attention layers, we use the global prompt for global shading."
        if is_cross and is_vanilla_cross:
            encoder_hidden_states = encoder_hidden_states[:2, ...]

        # In this case, we need to use MIGC or naive_fuser, so we copy the hidden_states_cond (instance_num+1) times for QKV
        if is_cross and not is_vanilla_cross:
            hidden_states_uncond = hidden_states[[0], ...]
            hidden_states_cond = hidden_states[[1], ...].repeat(instance_num + 1, 1, 1)
            hidden_states = torch.cat([hidden_states_uncond, hidden_states_cond])

        # QKV Operation of Vanilla Self-Attention or Cross-Attention
        query = attn.to_q(hidden_states)  # query此处为(2, HW, 320)
        q_ba = attn.to_q(hidden_states_ba)
        if (
                not is_cross
                and use_sa_preserve
                and timestep.item() in self.embedding
                and self.place_in_unet == "up"
        ):  # 这里是把unet的K和V concat起来，对应paper中的第7页的红字部分
            hidden_states = torch.cat(
                (hidden_states, torch.from_numpy(self.embedding[timestep.item()]).to(hidden_states.device)), dim=1)

        if not is_cross and sa_preserve and self.place_in_unet == "up":
            self.embedding[timestep.item()] = ori_hidden_states.cpu().numpy()

        # 为什么上面会有这么多的if判断？不是很明白

        encoder_hidden_states = (  # 这里是float16 # cross层就用encoder_hidden_states，self层就用hidden_states
            encoder_hidden_states
            if encoder_hidden_states is not None
            else hidden_states
        )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        query = attn.head_to_batch_dim(query)  # query永远来自self的hidden_states
        q_ba = attn.head_to_batch_dim(q_ba)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        if is_cross:
            k_ba = attn.to_k(encoder_hidden_states_ba)
            v_ba = attn.to_v(encoder_hidden_states_ba)
            k_ba = attn.head_to_batch_dim(k_ba)
            v_ba = attn.head_to_batch_dim(v_ba)
        else:
            k_ba = key
            v_ba = value
        attention_probs = attn.get_attention_scores(query, key, attention_mask)  # 48 4096 77
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        hidden_states = self.attnstore(attention_probs, is_cross, self.place_in_unet, q_ba, k_ba, v_ba, bboxes, 8, self.scale,
                       BaSteps, hidden_states)  # 正确做法是把q k v ith bboxes啥的塞到这里面去噢
        # q k v probs hidden_states全是float16
        ###### Self-Attention Results ######
        if not is_cross:
            return hidden_states  # 从这里返回了

        ###### Vanilla Cross-Attention Results ######
        if is_vanilla_cross:
            return hidden_states

        # 上述两个条件是不使用migc的时候，直接返回对应的值。下面才是重要的部分

        ###### Cross-Attention with MIGC ######
        assert (not is_vanilla_cross)
        # hidden_states: torch.Size([1+1+instance_num, HW, C]), the first 1 is the uncond ca output, the second 1 is the global ca output.
        # 看起来要先过几个module才到这里。这是paper最后的SAC(Shading Aggregation Controller)了
        hidden_states_uncond = hidden_states[[0], ...]  # torch.Size([1, HW, C])
        cond_ca_output = hidden_states[1:, ...].unsqueeze(0)  # torch.Size([1, 1+instance_num, 5, 64, 1280])
        guidance_masks = []
        in_box = []
        # Construct Instance Guidance Mask
        for bbox in bboxes[0]:
            guidance_mask = np.zeros((height, width))
            w_min = int(width * bbox[0])
            w_max = int(width * bbox[2])
            h_min = int(height * bbox[1])
            h_max = int(height * bbox[3])
            guidance_mask[h_min: h_max, w_min: w_max] = 1.0
            guidance_masks.append(guidance_mask[None, ...])  # guidance_mask有点硬了，确实得软一点
            in_box.append([bbox[0], bbox[2], bbox[1], bbox[3]])
        
        # Construct Background Guidance Mask
        sup_mask = get_sup_mask(guidance_masks)
        supplement_mask = torch.from_numpy(sup_mask[None, ...])
        # supplement_mask = F.interpolate(supplement_mask, (height//8, width//8), mode='bilinear').float()
        supplement_mask = F.interpolate(supplement_mask, (height // 8, width // 8), mode='bilinear').half()
        supplement_mask = supplement_mask.to(hidden_states.device)  # (1, 1, H, W)

        guidance_masks = np.concatenate(guidance_masks, axis=0)
        guidance_masks = guidance_masks[None, ...]
        # guidance_masks = torch.from_numpy(guidance_masks).float().to(cond_ca_output.device)
        guidance_masks = torch.from_numpy(guidance_masks).half().to(cond_ca_output.device)
        guidance_masks = F.interpolate(guidance_masks, (height // 8, width // 8),
                                       mode='bilinear')  # (1, instance_num, H, W)

        # in_box = torch.from_numpy(np.array(in_box))[None, ...].float().to(cond_ca_output.device)  # (1, instance_num, 4)
        in_box = torch.from_numpy(np.array(in_box))[None, ...].half().to(cond_ca_output.device)
        other_info = {}
        other_info['image_token'] = hidden_states_cond[None, ...]
        other_info['context'] = encoder_hidden_states[1:, ...]
        other_info['box'] = in_box
        other_info['context_pooler'] = embeds_pooler  # (instance_num, 1, 768)
        other_info['supplement_mask'] = supplement_mask
        other_info['attn2'] = None
        other_info['attn'] = attn
        other_info['height'] = height
        other_info['width'] = width
        other_info['ca_scale'] = ca_scale
        other_info['ea_scale'] = ea_scale
        other_info['sac_scale'] = sac_scale

        if not not_use_migc:
            hidden_states_cond, fuser_info = self.migc(cond_ca_output,
                                                       guidance_masks,
                                                       other_info=other_info,
                                                       return_fuser_info=True)
        else:  # 如果不使用migc，那么就和paper中说的一样用global prompt来控制
            hidden_states_cond, fuser_info = self.naive_fuser(cond_ca_output,
                                                              guidance_masks,  # 其h, w为input image的size // 8
                                                              other_info=other_info,
                                                              return_fuser_info=True)
        hidden_states_cond = hidden_states_cond.squeeze(1)

        hidden_states = torch.cat([hidden_states_uncond, hidden_states_cond])
        return hidden_states


class StableDiffusionMIGCPipeline(StableDiffusionPipeline):
    def __init__(
            self,
            vae: AutoencoderKL,
            text_encoder: CLIPTextModel,
            tokenizer: CLIPTokenizer,
            unet: UNet2DConditionModel,
            scheduler: KarrasDiffusionSchedulers,
            safety_checker: StableDiffusionSafetyChecker,
            feature_extractor: CLIPImageProcessor,
            image_encoder: CLIPVisionModelWithProjection = None,
            requires_safety_checker: bool = True,
    ):
        # Get the parameter signature of the parent class constructor
        parent_init_signature = inspect.signature(super().__init__)
        parent_init_params = parent_init_signature.parameters
        # 意思是从StableDiffusionPipeline中继承了vae, unet, textencoder等等参数，就是下面dict里的那些
        # Dynamically build a parameter dictionary based on the parameters of the parent class constructor
        # brushnet并没有构建参数字典？
        init_kwargs = {
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "unet": unet,
            "scheduler": scheduler,
            "safety_checker": safety_checker,
            "feature_extractor": feature_extractor,
            "requires_safety_checker": requires_safety_checker
        }  # 意思是把继承过来的参数变成一个dict，方便调用和记录。不过有个问题，在哪里把migc注入到unet的mid-block和deep up-block里呢？
        if 'image_encoder' in parent_init_params.items():
            init_kwargs['image_encoder'] = image_encoder
        super().__init__(**init_kwargs)

        self.instance_set = set()
        self.embedding = {}

    def _encode_prompt(
            self,
            prompts,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt=None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
             prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
        """
        if prompts is not None and isinstance(prompts, str):
            batch_size = 1
        elif prompts is not None and isinstance(prompts, list):
            batch_size = len(prompts)
        else:
            batch_size = prompt_embeds.shape[0]

        prompt_embeds_none_flag = (prompt_embeds is None)
        prompt_embeds_list = []
        embeds_pooler_list = []
        for prompt in prompts:
            if prompt_embeds_none_flag:
                # textual inversion: procecss multi-vector tokens if necessary
                if isinstance(self, TextualInversionLoaderMixin):
                    prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

                text_inputs = self.tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_input_ids = text_inputs.input_ids
                untruncated_ids = self.tokenizer(
                    prompt, padding="longest", return_tensors="pt"
                ).input_ids

                if untruncated_ids.shape[-1] >= text_input_ids.shape[
                    -1
                ] and not torch.equal(text_input_ids, untruncated_ids):
                    removed_text = self.tokenizer.batch_decode(
                        untruncated_ids[:, self.tokenizer.model_max_length - 1: -1]
                    )
                    logger.warning(
                        "The following part of your input was truncated because CLIP can only handle sequences up to"
                        f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                    )

                if (
                        hasattr(self.text_encoder.config, "use_attention_mask")
                        and self.text_encoder.config.use_attention_mask
                ):
                    attention_mask = text_inputs.attention_mask.to(device)
                else:
                    attention_mask = None

                prompt_embeds = self.text_encoder(
                    text_input_ids.to(device),
                    attention_mask=attention_mask,
                )
                embeds_pooler = prompt_embeds.pooler_output  # 这是比brushnet多出来的一行
                prompt_embeds = prompt_embeds[0]

            prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)
            # 下面的部分涉及到embeds_pooler的部分brushnet没有，差异开始突显
            embeds_pooler = embeds_pooler.to(dtype=self.text_encoder.dtype, device=device)

            bs_embed, seq_len, _ = prompt_embeds.shape
            # duplicate text embeddings for each generation per prompt, using mps friendly method  # 看不懂
            prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
            embeds_pooler = embeds_pooler.repeat(1, num_images_per_prompt)
            prompt_embeds = prompt_embeds.view(
                bs_embed * num_images_per_prompt, seq_len, -1
            )
            embeds_pooler = embeds_pooler.view(
                bs_embed * num_images_per_prompt, -1
            )
            prompt_embeds_list.append(prompt_embeds)
            embeds_pooler_list.append(embeds_pooler)
        prompt_embeds = torch.cat(prompt_embeds_list, dim=0)
        embeds_pooler = torch.cat(embeds_pooler_list, dim=0)
        # negative_prompt_embeds: (prompt_nums[0]+prompt_nums[1]+...prompt_nums[n], token_num, token_channel), <class 'torch.Tensor'>

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                negative_prompt = "worst quality, low quality, bad anatomy"
            uncond_tokens = [negative_prompt] * batch_size

            # textual inversion: process multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if (
                    hasattr(self.text_encoder.config, "use_attention_mask")
                    and self.text_encoder.config.use_attention_mask
            ):
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(
                dtype=self.text_encoder.dtype, device=device
            )

            negative_prompt_embeds = negative_prompt_embeds.repeat(
                1, num_images_per_prompt, 1
            )
            negative_prompt_embeds = negative_prompt_embeds.view(
                batch_size * num_images_per_prompt, seq_len, -1
            )
            # negative_prompt_embeds: (len(prompt_nums), token_num, token_channel), <class 'torch.Tensor'>

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            final_prompt_embeds = torch.cat(
                [negative_prompt_embeds, prompt_embeds])  # 这是Migc的特殊之处。需要搞清楚为什么need to do two forward passes?

        return final_prompt_embeds, prompt_embeds, embeds_pooler[:, None, :]

    def check_inputs(
            self,
            prompt,
            token_indices,
            bboxes,
            height,
            width,
            callback_steps,
            negative_prompt=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
            )

        if (callback_steps is None) or (
                callback_steps is not None
                and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (
                not isinstance(prompt, str) and not isinstance(prompt, list)
        ):
            raise ValueError(
                f"`prompt` has to be of type `str` or `list` but is {type(prompt)}"
            )

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        if token_indices is not None:
            if isinstance(token_indices, list):
                if isinstance(token_indices[0], list):
                    if isinstance(token_indices[0][0], list):
                        token_indices_batch_size = len(token_indices)
                    elif isinstance(token_indices[0][0], int):
                        token_indices_batch_size = 1
                    else:
                        raise TypeError(
                            "`token_indices` must be a list of lists of integers or a list of integers."
                        )
                else:
                    raise TypeError(
                        "`token_indices` must be a list of lists of integers or a list of integers."
                    )
            else:
                raise TypeError(
                    "`token_indices` must be a list of lists of integers or a list of integers."
                )

        if bboxes is not None:
            if isinstance(bboxes, list):
                if isinstance(bboxes[0], list):
                    if (
                            isinstance(bboxes[0][0], list)
                            and len(bboxes[0][0]) == 4
                            and all(isinstance(x, float) for x in bboxes[0][0])
                    ):
                        bboxes_batch_size = len(bboxes)
                    elif (
                            isinstance(bboxes[0], list)
                            and len(bboxes[0]) == 4
                            and all(isinstance(x, float) for x in bboxes[0])
                    ):
                        bboxes_batch_size = 1
                    else:
                        print(isinstance(bboxes[0], list), len(bboxes[0]))
                        raise TypeError(
                            "`bboxes` must be a list of lists of list with four floats or a list of tuples with four floats."
                        )  # bboxes就是xyxy
                else:
                    print(isinstance(bboxes[0], list), len(bboxes[0]))
                    raise TypeError(
                        "`bboxes` must be a list of lists of list with four floats or a list of tuples with four floats."
                    )
            else:
                print(isinstance(bboxes[0], list), len(bboxes[0]))
                raise TypeError(
                    "`bboxes` must be a list of lists of list with four floats or a list of tuples with four floats."
                )

        if prompt is not None and isinstance(prompt, str):
            prompt_batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            prompt_batch_size = len(prompt)
        elif prompt_embeds is not None:
            prompt_batch_size = prompt_embeds.shape[0]

        if token_indices_batch_size != prompt_batch_size:
            raise ValueError(
                f"token indices batch size must be same as prompt batch size. token indices batch size: {token_indices_batch_size}, prompt batch size: {prompt_batch_size}"
            )

        if bboxes_batch_size != prompt_batch_size:
            raise ValueError(
                f"bbox batch size must be same as prompt batch size. bbox batch size: {bboxes_batch_size}, prompt batch size: {prompt_batch_size}"
            )

    def get_indices(self, prompt: str) -> Dict[str, int]:
        """Utility function to list the indices of the tokens you wish to alte"""
        ids = self.tokenizer(prompt).input_ids
        indices = {
            i: tok
            for tok, i in zip(
                self.tokenizer.convert_ids_to_tokens(ids), range(len(ids))
            )
        }
        return indices

    @staticmethod
    def draw_box(pil_img: Image, bboxes: List[List[float]]) -> Image:
        """Utility function to draw bbox on the image"""
        width, height = pil_img.size
        draw = ImageDraw.Draw(pil_img)  # 调的包

        for obj_box in bboxes:
            x_min, y_min, x_max, y_max = (  # bboxes里面的参数就是点的坐标所占width和height的百分比，参考mig_bench_anno.yaml
                obj_box[0] * width,
                obj_box[1] * height,
                obj_box[2] * width,
                obj_box[3] * height,
            )
            draw.rectangle(
                [int(x_min), int(y_min), int(x_max), int(y_max)],
                outline="red",
                width=4,
            )

        return pil_img

    @staticmethod
    def draw_box_desc(pil_img: Image, bboxes: List[List[float]], prompt: List[str]) -> Image:
        """Utility function to draw bbox on the image"""
        color_list = ['red', 'blue', 'yellow', 'purple', 'green', 'black', 'brown', 'orange', 'white', 'gray']
        width, height = pil_img.size
        draw = ImageDraw.Draw(pil_img)
        font_folder = os.path.dirname(os.path.dirname(__file__))
        font_path = os.path.join(font_folder, 'Rainbow-Party-2.ttf')
        font = ImageFont.truetype(font_path, 30)

        for box_id in range(len(bboxes)):
            obj_box = bboxes[box_id]
            text = prompt[box_id]
            fill = 'black'  # the black box represents that the instance does not have a specified color.
            for color in prompt[box_id].split(' '):
                if color in color_list:
                    fill = color
            text = text.split(',')[0]
            x_min, y_min, x_max, y_max = (
                obj_box[0] * width,
                obj_box[1] * height,
                obj_box[2] * width,
                obj_box[3] * height,
            )
            draw.rectangle(
                [int(x_min), int(y_min), int(x_max), int(y_max)],
                outline=fill,
                width=4,
            )
            draw.text((int(x_min), int(y_min)), text, fill=fill, font=font)

        return pil_img

    @torch.no_grad()
    def __call__(  # 开始执行主函数
            self,
            prompt: List[List[str]] = None,
            bboxes: List[List[List[float]]] = None,
            height: Optional[int] = None,  # migc这里也是None？看来和brushnet是一样的，之前搞错了
            width: Optional[int] = None,
            num_inference_steps: int = 50,  # 这个和brushnet一致
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: int = 1,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            MIGCsteps=20,
            NaiveFuserSteps=-1,
            ca_scale=None,
            ea_scale=None,
            sac_scale=None,
            aug_phase_with_and=False,
            sa_preserve=False,
            use_sa_preserve=False,
            clear_set=False,
            GUI_progress=None
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            token_indices (Union[List[List[List[int]]], List[List[int]]], optional):
                The list of the indexes in the prompt to layout. Defaults to None.
            bboxes (Union[List[List[List[float]]], List[List[float]]], optional):
                The bounding boxes of the indexes to maintain layout in the image. Defaults to None.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
            max_guidance_iter (`int`, *optional*, defaults to `10`):
                The maximum number of iterations for the layout guidance on attention maps in diffusion mode.
            max_guidance_iter_per_step (`int`, *optional*, defaults to `5`):
                The maximum number of iterations to run during each time step for layout guidance.
            scale_factor (`int`, *optional*, defaults to `50`):
                The scale factor used to update the latents during optimization.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """

        def aug_phase_with_and_function(phase, instance_num):
            instance_num = min(instance_num, 7)
            copy_phase = [phase] * instance_num
            phase = ', and '.join(copy_phase)
            return phase

        if aug_phase_with_and:
            instance_num = len(prompt[0]) - 1
            for i in range(1, len(prompt[0])):
                prompt[0][i] = aug_phase_with_and_function(prompt[0][i],
                                                           instance_num)
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        prompt_nums = [0] * len(prompt)  # 这个[0]有点抽象，先标记一手
        for i, _ in enumerate(prompt):
            prompt_nums[i] = len(_)

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds, cond_prompt_embeds, embeds_pooler = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )
        # print(prompt_embeds.shape)  3 77 768

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)  # 需要看看extra_step_kwargs是什么？

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        if clear_set:
            self.instance_set = set()
            self.embedding = {}

        now_set = set()
        for i in range(len(bboxes[0])):
            now_set.add((tuple(bboxes[0][i]), prompt[0][i + 1]))

        mask_set = (now_set | self.instance_set) - (now_set & self.instance_set)
        self.instance_set = now_set

        guidance_mask = np.full((4, height // 8, width // 8), 1.0)

        for bbox, _ in mask_set:  # 这个5是什么意思？为什么会是5？
            w_min = max(0, int(width * bbox[0] // 8) - 5)
            w_max = min(width, int(width * bbox[2] // 8) + 5)
            h_min = max(0, int(height * bbox[1] // 8) - 5)
            h_max = min(height, int(height * bbox[3] // 8) + 5)
            guidance_mask[:, h_min:h_max, w_min:w_max] = 0

        kernal_size = 5
        guidance_mask = uniform_filter(
            guidance_mask, axes=(1, 2), size=kernal_size
        )

        guidance_mask = torch.from_numpy(guidance_mask).to(self.device).unsqueeze(0)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if GUI_progress is not None:
                    GUI_progress[0] = int((i + 1) / len(timesteps) * 100)
                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                # predict the noise residual
                cross_attention_kwargs = {'prompt_nums': prompt_nums,
                                          'bboxes': bboxes,
                                          'ith': i,
                                          'embeds_pooler': embeds_pooler,
                                          'timestep': t,
                                          'height': height,
                                          'width': width,
                                          'MIGCsteps': MIGCsteps,
                                          'NaiveFuserSteps': NaiveFuserSteps,
                                          'ca_scale': ca_scale,
                                          'ea_scale': ea_scale,
                                          'sac_scale': sac_scale,
                                          'sa_preserve': sa_preserve,
                                          'use_sa_preserve': use_sa_preserve}

                self.unet.eval()
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                            noise_pred_text - noise_pred_uncond
                    )

                step_output = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs
                )
                latents = step_output.prev_sample

                ori_input = latents.detach().clone()
                if use_sa_preserve and i in self.embedding:
                    latents = (
                            latents * (1.0 - guidance_mask)
                            + torch.from_numpy(self.embedding[i]).to(latents.device) * guidance_mask
                    ).float()

                if sa_preserve:
                    self.embedding[i] = ori_input.cpu().numpy()

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                        (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        if output_type == "latent":
            image = latents
        elif output_type == "pil":
            # 8. Post-processing
            image = self.decode_latents(latents)
            image = self.numpy_to_pil(image)
        else:
            # 8. Post-processing
            image = self.decode_latents(latents)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, None)

        return StableDiffusionPipelineOutput(
            images=image, nsfw_content_detected=None
        )