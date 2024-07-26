import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Union, Tuple, List, Callable, Dict

# from torchvision.utils import save_image
from einops import rearrange, repeat


class AttentionBase:
    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

    def before_step(self):
        pass

    def after_step(self):
        pass

    def __call__(self, q, k, v, is_cross, place_in_unet, num_heads, **kwargs):
        if self.cur_att_layer == 0:
            self.before_step()

        out = self.forward(q, k, v, is_cross, place_in_unet, num_heads, **kwargs)  # 从这里进入Bounded_attention的forward函数
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.after_step()

        return out

    def forward(self, q, k, v, is_cross, place_in_unet, num_heads, **kwargs):  # 这是为了继承而写的父类，不会调用
        batch_size = q.size(0) // num_heads
        n = q.size(1)
        d = k.size(1)

        q = q.reshape(batch_size, num_heads, n, -1)
        k = k.reshape(batch_size, num_heads, d, -1)
        v = v.reshape(batch_size, num_heads, d, -1)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=kwargs['mask'])
        out = out.reshape(batch_size * num_heads, n, -1)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=num_heads)
        return out

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0


def register_attention_editor_diffusers(model, editor: AttentionBase):
    """
    Register a attention editor to Diffuser Pipeline, refer from [Prompt-to-Prompt]
    """
    def ca_forward(self, place_in_unet):  # 把这里的forward改成migcprocessor的__call__
        def forward(x, encoder_hidden_states=None, attention_mask=None, context=None, mask=None, ith=None):  # cao，这里的x就是hidden_states
            """
            The attention is similar to the original implementation of LDM CrossAttention class
            except adding some modifications on the attention
            """
            print(ith)
            if encoder_hidden_states is not None:
                context = encoder_hidden_states
            if attention_mask is not None:
                mask = attention_mask

            to_out = self.to_out
            if isinstance(to_out, nn.modules.container.ModuleList):
                to_out = self.to_out[0]
            else:
                to_out = self.to_out

            h = self.heads
            q = self.to_q(x)
            is_cross = context is not None
            context = context if is_cross else x
            k = self.to_k(context)
            v = self.to_v(context)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
            out = editor(  #
                q, k, v, is_cross, place_in_unet,
                self.heads, scale=self.scale, mask=mask)  # 从这里进入attentionbase的__call__函数

            return to_out(out)

        return forward

    def register_editor(net, count, place_in_unet):
        for name, subnet in net.named_children():
            if net.__class__.__name__ == 'Attention':  # spatial Transformer layer
                net.original_forward = net.forward
                net.forward = ca_forward(net, place_in_unet)  # 关键在这里，把attention层的forward修改掉，而不是像migc一样仅修改attn_processors
                return count + 1        # 所以如果我想把migc改成ba这样，就必须要修改unet的forward函数噢
            elif hasattr(net, 'children'):
                count = register_editor(subnet, count, place_in_unet)
        return count

    cross_att_count = 0  # 这个和migc一样是直接改变unet的，估计也是从pipe里的unet进入bounded attention
    for net_name, net in model.unet.named_children():
        if "down" in net_name:
            cross_att_count += register_editor(net, 0, "down")
        elif "mid" in net_name:
            cross_att_count += register_editor(net, 0, "mid")
        elif "up" in net_name:
            cross_att_count += register_editor(net, 0, "up")

    editor.num_att_layers = cross_att_count
    editor.model = model  # 这个和Load_migc没啥区别，都是把方法注入到32个attention layer里面
    model.editor = editor

