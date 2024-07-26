import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from migc.migc_layers import CBAM, CrossAttention, LayoutAttention
from einops.einops import rearrange
from torch_kmeans import KMeans

class FourierEmbedder():
    def __init__(self, num_freqs=64, temperature=100):
        self.num_freqs = num_freqs
        self.temperature = temperature
        self.freq_bands = temperature ** ( torch.arange(num_freqs) / num_freqs )

    @ torch.no_grad()
    def __call__(self, x, cat_dim=-1):
        out = []
        for freq in self.freq_bands:
            out.append( torch.sin( freq*x ) )
            out.append( torch.cos( freq*x ) )
        return torch.cat(out, cat_dim)  # torch.Size([5, 30, 64])


class PositionNet(nn.Module):
    def __init__(self, in_dim, out_dim, fourier_freqs=8):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.fourier_embedder = FourierEmbedder(num_freqs=fourier_freqs)
        self.position_dim = fourier_freqs * 2 * 4  # 2 is sin&cos, 4 is xyxy

        # -------------------------------------------------------------- #
        self.linears_position = nn.Sequential(
            nn.Linear(self.position_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, out_dim),
        )

    def forward(self, boxes):

        # embedding position (it may includes padding as placeholder)
        xyxy_embedding = self.fourier_embedder(boxes)  # B*1*4 --> B*1*C torch.Size([5, 1, 64])
        xyxy_embedding = self.linears_position(xyxy_embedding)  # B*1*C --> B*1*768 torch.Size([5, 1, 768])

        return xyxy_embedding


class SAC(nn.Module):
    def __init__(self, C, number_pro=30):
        super().__init__()
        self.C = C
        self.number_pro = number_pro
        self.conv1 = nn.Conv2d(C + 1, C, 1, 1)
        self.cbam1 = CBAM(C)
        self.conv2 = nn.Conv2d(C, 1, 1, 1)
        self.cbam2 = CBAM(number_pro, reduction_ratio=1)

    def forward(self, x, guidance_mask, sac_scale=None):
        '''
        :param x: (B, phase_num, HW, C)
        :param guidance_mask: (B, phase_num, H, W)
        :return:
        '''
        B, phase_num, HW, C = x.shape
        _, _, H, W = guidance_mask.shape
        guidance_mask = guidance_mask.view(guidance_mask.shape[0], phase_num, -1)[
            ..., None]  # (B, phase_num, HW, 1)

        null_x = torch.zeros_like(x[:, [0], ...]).to(x.device)
        null_mask = torch.zeros_like(guidance_mask[:, [0], ...]).to(guidance_mask.device)

        x = torch.cat([x, null_x], dim=1)
        guidance_mask = torch.cat([guidance_mask, null_mask], dim=1)
        phase_num += 1


        scale = torch.cat([x, guidance_mask], dim=-1)  # (B, phase_num, HW, C+1)
        scale = scale.view(-1, H, W, C + 1)  # (B * phase_num, H, W, C+1)
        scale = scale.permute(0, 3, 1, 2)  # (B * phase_num, C+1, H, W)
        scale = self.conv1(scale)  # (B * phase_num, C, H, W)
        scale = self.cbam1(scale)  # (B * phase_num, C, H, W)
        scale = self.conv2(scale)  # (B * phase_num, 1, H, W)
        scale = scale.view(B, phase_num, H, W)  # (B, phase_num, H, W)

        null_scale = scale[:, [-1], ...]
        scale = scale[:, :-1, ...]
        x = x[:, :-1, ...]

        pad_num = self.number_pro - phase_num + 1

        ori_phase_num = scale[:, 1:-1, ...].shape[1]
        phase_scale = torch.cat([scale[:, 1:-1, ...], null_scale.repeat(1, pad_num, 1, 1)], dim=1)
        shuffled_order = torch.randperm(phase_scale.shape[1])
        inv_shuffled_order = torch.argsort(shuffled_order)

        random_phase_scale = phase_scale[:, shuffled_order, ...]

        scale = torch.cat([scale[:, [0], ...], random_phase_scale, scale[:, [-1], ...]], dim=1)
        # (B, number_pro, H, W)

        scale = self.cbam2(scale)  # (B, number_pro, H, W)
        scale = scale.view(B, self.number_pro, HW)[..., None]  # (B, number_pro, HW)

        random_phase_scale = scale[:, 1: -1, ...]
        phase_scale = random_phase_scale[:, inv_shuffled_order[:ori_phase_num], :]
        if sac_scale is not None:
            instance_num = len(sac_scale)
            for i in range(instance_num):
                phase_scale[:, i, ...] = phase_scale[:, i, ...] * sac_scale[i]


        scale = torch.cat([scale[:, [0], ...], phase_scale, scale[:, [-1], ...]], dim=1)

        scale = scale.softmax(dim=1)  # (B, phase_num, HW, 1)
        out = (x * scale).sum(dim=1, keepdims=True)  # (B, 1, HW, C)
        return out, scale


class MIGC(nn.Module):  # 这就是我要找的东西，把它去替换brushnet中的frozen unet即可！
    def __init__(self, C, attn_type='base', context_dim=768, heads=8):
        super().__init__()
        self.ea = CrossAttention(query_dim=C, context_dim=context_dim,
                             heads=heads, dim_head=C // heads,
                             dropout=0.0)
        self.la = LayoutAttention(query_dim=C,
                                    heads=heads, dim_head=C // heads,
                                    dropout=0.0)
        self.norm = nn.LayerNorm(C)
        self.sac = SAC(C)
        self.pos_net = PositionNet(in_dim=768, out_dim=768)

    def forward(self, ca_x, guidance_mask, other_info, return_fuser_info=False):
        # x: (B, instance_num+1, HW, C)
        # guidance_mask: (B, instance_num, H, W)
        # box: (instance_num, 4)
        # image_token: (B, instance_num+1, HW, C)
        full_H = other_info['height']
        full_W = other_info['width']
        B, _, HW, C = ca_x.shape
        instance_num = guidance_mask.shape[1]
        down_scale = int(math.sqrt(full_H * full_W // ca_x.shape[2]))
        H = full_H // down_scale
        W = full_W // down_scale
        guidance_mask = F.interpolate(guidance_mask, size=(H, W), mode='bilinear')   # (B, instance_num, H, W)


        supplement_mask = other_info['supplement_mask']  # (B, 1, 64, 64)
        supplement_mask = F.interpolate(supplement_mask, size=(H, W), mode='bilinear')  # (B, 1, H, W)
        image_token = other_info['image_token']
        assert image_token.shape == ca_x.shape
        context = other_info['context_pooler']
        box = other_info['box']
        box = box.view(B * instance_num, 1, -1)
        box_token = self.pos_net(box)
        context = torch.cat([context[1:, ...], box_token], dim=1)
        ca_scale = other_info['ca_scale'] if 'ca_scale' in other_info else None
        ea_scale = other_info['ea_scale'] if 'ea_scale' in other_info else None
        sac_scale = other_info['sac_scale'] if 'sac_scale' in other_info else None

        ea_x, ea_attn = self.ea(self.norm(image_token[:, 1:, ...].view(B * instance_num, HW, C)),
                                     context=context, return_attn=True)
        ea_x = ea_x.view(B, instance_num, HW, C)
        ea_x = ea_x * guidance_mask.view(B, instance_num, HW, 1)

        ca_x[:, 1:, ...] = ca_x[:, 1:, ...] * guidance_mask.view(B, instance_num, HW, 1)  # (B, phase_num, HW, C)
        if ca_scale is not None:
            assert len(ca_scale) == instance_num
            for i in range(instance_num):
                ca_x[:, i+1, ...] = ca_x[:, i+1, ...] * ca_scale[i] + ea_x[:, i, ...] * ea_scale[i]
        else:
            ca_x[:, 1:, ...] = ca_x[:, 1:, ...] + ea_x

        ori_image_token = image_token[:, 0, ...]  # (B, HW, C)
        fusion_template = self.la(x=ori_image_token, guidance_mask=torch.cat([guidance_mask[:, :, ...], supplement_mask], dim=1))  # (B, HW, C)
        fusion_template = fusion_template.view(B, 1, HW, C)  # (B, 1, HW, C)

        ca_x = torch.cat([ca_x, fusion_template], dim = 1)
        ca_x[:, 0, ...] = ca_x[:, 0, ...] * supplement_mask.view(B, HW, 1)
        guidance_mask = torch.cat([
            supplement_mask,
            guidance_mask, 
            torch.ones(B, 1, H, W).to(guidance_mask.device)
            ], dim=1)


        out_MIGC, sac_scale = self.sac(ca_x, guidance_mask, sac_scale=sac_scale)
        if return_fuser_info:
            fuser_info = {}
            fuser_info['sac_scale'] = sac_scale.view(B, instance_num + 2, H, W)
            fuser_info['ea_attn'] = ea_attn.mean(dim=1).view(B, instance_num, H, W, 2)
            return out_MIGC, fuser_info
        else:
            return out_MIGC


class NaiveFuser(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, ca_x, guidance_mask, other_info, return_fuser_info=False):
        # ca_x: (B, instance_num+1, HW, C)
        # guidance_mask: (B, instance_num, H, W)
        # box: (instance_num, 4)
        # image_token: (B, instance_num+1, HW, C)
        full_H = other_info['height']
        full_W = other_info['width']
        B, _, HW, C = ca_x.shape  # hw = guidance_mask的 h * w
        instance_num = guidance_mask.shape[1]
        down_scale = int(math.sqrt(full_H * full_W // ca_x.shape[2]))
        H = full_H // down_scale
        W = full_W // down_scale
        guidance_mask = F.interpolate(guidance_mask, size=(H, W), mode='bilinear')   # (B, instance_num, H, W)
        guidance_mask = torch.cat([torch.ones(B, 1, H, W).to(guidance_mask.device), guidance_mask * 10], dim=1)  # (B, instance_num+1, H, W)
        guidance_mask = guidance_mask.view(B, instance_num + 1, HW, 1)
        out_MIGC = (ca_x * guidance_mask).sum(dim=1) / (guidance_mask.sum(dim=1) + 1e-6)
        if return_fuser_info:
            return out_MIGC, None
        else:
            return out_MIGC
class BaMask(nn.Module):
    EPSILON = 1e-5
    def __init__(
        self,
        scale,
        # boxes,
        # prompts,
        # subject_token_indices,
        # cross_loss_layers,
        # self_loss_layers,
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
        delta_refine_mask_steps=5,
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
        super().__init__()
        self.scale = scale
        self.cross_mask_layers = {14, 15, 16, 17, 18, 19}
        self.self_mask_layers = {14, 15, 16, 17, 18, 19}
        self.eos_token_index = eos_token_index
        self.filter_token_indices = filter_token_indices
        self.leading_token_indices = leading_token_indices
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
    def forward(self, hidden_states, bboxes, q, k, v, is_cross, ith, num_heads=8, place_in_unet=None, **kwargs):
        judge = q.shape == torch.Size([16, 256, 160])
        batch_size = q.size(0) // num_heads  # 用的是forward而不是__call__!
        n = q.size(1)
        d = k.size(1)
        dtype = q.dtype
        device = q.device
        if is_cross:
            return hidden_states
            # masks = self._hide_other_subjects_from_tokens(batch_size // 2, ith, bboxes, n, d, dtype, device)
        else:
            masks = self._hide_other_subjects_from_subjects(batch_size // 2, ith, bboxes, n, dtype, device)
        # 这里的masks是attention的masks，所以size为(2, 1, 64, 64)。但migc并没有用到attention mask?得再确定一下
        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale  # 这是做scale-dot production
        sim = sim.reshape(batch_size, num_heads, n, d) + masks
        attn = sim.reshape(-1, n, d).softmax(-1)
        self._save(attn, is_cross, num_heads, place_in_unet, judge)
        out = torch.bmm(attn, v)
        hidden_states = rearrange(out, '(b h) n d -> b n (h d)', h=num_heads)
        return hidden_states
    # 下面是_save部分相关函数
    def _save(self, attn, is_cross, num_heads, place_in_unet, judge):
        _, attn = attn.chunk(2)
        attn = attn.reshape(-1, num_heads, *attn.shape[-2:])  # b h n k

        self._save_mask_maps(attn, is_cross, place_in_unet, judge)

    def _save_mask_maps(self, attn, is_cross, place_in_unet, judge):
        if (
                (self.optimized) or
                (is_cross and not (place_in_unet == 'up' and judge)) or
                ((not is_cross) and (not (place_in_unet == 'up' and judge)))
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
    def _hide_other_subjects_from_tokens(self, batch_size, ith, bboxes, n, d, dtype, device):  # b h i j
        resolution = int(n ** 0.5)
        subject_masks, background_masks = self._obtain_masks(resolution, ith, bboxes, batch_size=batch_size, device=device)  # b s n
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

    def _hide_other_subjects_from_subjects(self, batch_size, ith, bboxes, n, dtype, device):  # b h i j
        resolution = int(n ** 0.5)
        subject_masks, background_masks = self._obtain_masks(resolution, ith, bboxes, batch_size=batch_size, device=device)  # b s n
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
    def _obtain_masks(self, resolution, ith, bboxes, return_boxes=False, return_existing=False, batch_size=None, device=None):
        return_boxes = return_boxes or (return_existing and self.self_masks is None)
        # 这里就是开始refine mask的步骤！当return_boxes = False且现在的步数大于最小聚类步数
        if return_boxes or ith < 1:
            masks = self._convert_boxes_to_masks(resolution, bboxes, device=device).unsqueeze(0)
            if batch_size is not None:
                masks = masks.expand(batch_size, *masks.shape[1:])
        else:  # 这里就是开始refine mask的步骤！当return_boxes = False且现在的步数大于最小聚类步数
            masks = self._obtain_self_masks(resolution, ith, bboxes, return_existing=return_existing)  # 获得聚类后的mask后再返回
            if device is not None:
                masks = masks.to(device=device)

        background_mask = masks.sum(dim=1) == 0
        return masks, background_mask

    def _convert_boxes_to_masks(self, resolution, bboxes, device=None):  # s n
        boxes = torch.zeros(len(bboxes), resolution, resolution, dtype=bool, device=device)
        for i, box in enumerate(bboxes[0]):
            x0, x1 = box[0] * resolution, box[2] * resolution
            y0, y1 = box[1] * resolution, box[3] * resolution

            boxes[i, round(y0): round(y1), round(x0): round(x1)] = True

        return boxes.flatten(start_dim=1)

    def _obtain_self_masks(self, resolution, ith, bboxes, return_existing=False):
        if (
                (self.self_masks is None)
        ):
            self.self_masks = self._fix_zero_masks(self._build_self_masks(ith, bboxes), bboxes)  # 这是每个ith都修整mask

        b, s, n = self.self_masks.shape
        mask_resolution = int(n ** 0.5)
        self_masks = self.self_masks.reshape(b, s, mask_resolution, mask_resolution).float()
        self_masks = F.interpolate(self_masks, resolution, mode='nearest-exact')
        return self_masks.flatten(start_dim=2).bool()

    # build_self_masks更像是生成sa mask的函数！
    def _build_self_masks(self, ith, bboxes):
        c, clusters = self._cluster_self_maps()  # b n
        cluster_masks = torch.stack([(clusters == cluster_index) for cluster_index in range(c)], dim=2)  # b n c
        cluster_area = cluster_masks.sum(dim=1, keepdim=True)  # b 1 c

        n = clusters.size(1)
        resolution = int(n ** 0.5)
        cross_masks = self._obtain_cross_masks(resolution, ith, bboxes)  # b s n
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
        self._save_maps(self_masks, 'self_masks')
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
        self._save_maps(clusters / num_clusters, f'clusters')
        return num_clusters, clusters

    def _obtain_cross_masks(self, resolution, ith, bboxes, scale=10):
        maps = self._compute_maps(self.mean_cross_map, resolution=resolution)  # b n k
        maps = F.sigmoid(scale * (maps - self.cross_mask_threshold))
        maps = self._normalize_maps(maps, reduce_min=True)
        maps = maps.transpose(1, 2)  # b k n
        existing_masks, _ = self._obtain_masks(
            resolution, ith, bboxes, return_existing=True, batch_size=maps.size(0), device=maps.device)
        maps = maps * existing_masks.to(dtype=maps.dtype)
        self._save_maps(maps, 'cross_masks')
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
