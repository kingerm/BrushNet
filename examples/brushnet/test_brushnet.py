import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 本地改镜像站才能调试
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from diffusers import StableDiffusionBrushNetPipeline, BrushNetModel, UniPCMultistepScheduler, EulerDiscreteScheduler
import torch    # brushnet使用的是UniPCMultistepScheduler，而migc是EulerDiscreteScheduler，这是一个矛盾
import cv2
import numpy as np
from PIL import Image
# 下面import migc相关代码
from migc.migc_utils import seed_everything
from migc.migc_pipeline import StableDiffusionMIGCPipeline, MIGCProcessor, AttentionStore
from migc.migc_utils import load_migc
from torchvision.ops import masks_to_boxes
import math
# 下面是zone相关代码
import argparse
from mask_utils import *
from PIL import ImageEnhance

def read_mask(mask_path, kernel, iterations):
    name = mask_path[4: -4]
    mask = cv2.imread(mask_path)
    mask_image = 1.*(cv2.imread(mask_path).sum(-1)>255)[:,:,np.newaxis]
    dilated_mask_image = cv2.dilate(mask_image, kernel, iterations)[..., np.newaxis]
    mask_image_t = torch.from_numpy(mask_image).permute(2, 0, 1)
    box_xyxy = masks_to_boxes(mask_image_t)
    h, w, _ = mask_image.shape
    box_xyxy = box_xyxy.squeeze(0)
    box_xyxy[0], box_xyxy[2] = box_xyxy[0] / w, box_xyxy[2] / w
    box_xyxy[1], box_xyxy[3] = box_xyxy[1] / h, box_xyxy[3] / h
    return mask_image, box_xyxy.numpy().tolist(), name, dilated_mask_image, mask
def cal_hw(h, w):
    ref_aspect_ratio = h / w  # 定义参考的高宽比，目的是使变形后的size的高宽比尽可能接近原来的高宽比
    # 考虑到这样一个事实：对于正整数a而言，其相邻的两个32的倍数必然是一个除64余0， 一个除64余32.
    # 对于a/b而言，较小的一方在改变相同值时，会对比例产生更大的影响。所以应该尽可能地让较小的一方改变较少的值。
    dec_h, dec_w = (h % 32) / 32, (w % 32) / 32  # 计算h, w除以32的小数部分
    if h < w:
        int_h = h // 32 if dec_h < 0.5 else (h // 32) + 1
        h_new = int_h * 32
        tmp = (h_new % 64) / 64
        if tmp == 0.5:
            w_new = w // 64 * 64 + 32
        elif tmp == 0.0:
            w_new1 = w // 64 * 64
            w_new2 = w_new1 + 64
            asp1 = h_new / w_new1
            asp2 = h_new / w_new2
            if abs(asp1 - ref_aspect_ratio) < abs(asp2 - ref_aspect_ratio):
                w_new = w_new1
            else:
                w_new = w_new2
    else:
        int_w = w // 32 if dec_w < 0.5 else (w // 32) + 1
        w_new = int_w * 32
        tmp = (w_new % 64) / 64
        if tmp == 0.5:
            h_new = h // 64 * 64 + 32
        elif tmp == 0.0:
            h_new1 = h // 64 * 64
            h_new2 = h_new1 + 64
            asp1 = h_new1 / w_new
            asp2 = h_new2 / w_new
            if abs(asp1 - ref_aspect_ratio) < abs(asp2 - ref_aspect_ratio):
                h_new = h_new1
            else:
                h_new = h_new2
    return int(h_new), int(w_new)

def zone_ops(style_image, original_image, mask, args, sam_predictor):
    """

    Args:
        style_image: sd output image
        original_image: resized image or edited zone_ops image
        mask: input mask
        args: parser's args

    Returns: edited zone_ops image

    """

    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_CUBIC)  # 对输入的mask进行resize
    masked_out, mask = get_finemask_everything(mask, style_image,
                                               sam_predictor)  # masked_out是被mask后的style_image，mask是优化后的Mask
    # 这里不反色，会变成蓝色的
    # masked_out = cv2.cvtColor(masked_out, cv2.COLOR_BGR2RGB)
    # cv2.imwrite('masked_out1.png', masked_out)
    mask_image = mask

    # dilated_mask_all, _ = dilate_mask(mask_image, iterations=args.dilate_strength)  # 确实需要dilate，因为物体不一定能完全和想要的区域吻合
    # 7. ops based on mode
    # style_image_np = np.array(style_image)
    style_image_np = cv2.cvtColor(np.array(style_image), cv2.COLOR_BGR2RGB)
    gt_image_np = original_image  # original_image由于是直接imread的，所以不用反色
    # gt_image_np = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    # canvas = gt_image_np.copy()
    #
    # canvas[mask_image > 0] = masked_out[dilated_mask_all > 0]  # 这里可以把mask换成dilate_mask_all
    mask_blurred = cv2.GaussianBlur(mask_image, (11, 11), 0)
    mask_blurred = mask_blurred[..., np.newaxis] / 255
    new_canvas = gt_image_np * (1 - mask_blurred) + style_image_np * mask_blurred
    # cv2.imwrite('new_canvas.png', new_canvas)
    # edge_bimask = dilated_mask_all - mask_image
    # kernel = np.ones((3, 3), np.uint8)  # 增大闭运算的kernel size，可以优化边缘连接效果。闭运算可以填补小黑洞
    # edge_bimask = cv2.morphologyEx(edge_bimask, cv2.MORPH_OPEN, kernel)
    # cv2.imwrite('edge_bimask.png', edge_bimask)
    # edge_bimask_blurred = cv2.GaussianBlur(edge_bimask, (5, 5), 0)
    # cv2.imwrite('edge_bimask_blurred.png', edge_bimask)
    # edge_bimask_blurred = edge_bimask_blurred[..., np.newaxis] / 255
    # new_canvas = new_canvas * (1 - edge_bimask_blurred) + gt_image_np * edge_bimask_blurred
    # new_canvas = gt_image_np.copy()
    # new_canvas[mask_image > 0] = style_image_np[mask_image > 0]
    # new_canvas[edge_bimask > 0] = style_image_np[edge_bimask > 0] * args.alpha + gt_image_np[edge_bimask > 0] * (
    #         1 - args.alpha)
    # image_np = np.array(image)
    # init_image_np = cv2.imread(image_path)[:, :, ::-1]
    # mask_np = 1. * (cv2.imread(mask_path).sum(-1) > 255)[:, :, np.newaxis]
    #
    # # blur, you can adjust the parameters for better performance
    # mask_blurred = cv2.GaussianBlur(mask_np * 255, (21, 21), 0) / 255
    # mask_blurred = mask_blurred[:, :, np.newaxis]
    # mask_np = 1 - (1 - mask_np) * (1 - mask_blurred)
    #
    # image_pasted = init_image_np * (1 - mask_np) + image_np * mask_np
    # image_pasted = image_pasted.astype(image_np.dtype)
    # image = Image.fromarray(image_pasted)
    return new_canvas
def zone_parse_args():
    parser = argparse.ArgumentParser(description='Process some parameters.')
    parser.add_argument('--sam_ckpt_path', type=str, default="./ckpts/sam_vit_h_4b8939.pth",
                        help='The path to SAM checkpoint')
    parser.add_argument('--threshold', type=int, default=20, help='The threshold of edge smoother')
    parser.add_argument('--alpha', type=float, default=0.3,
                        help='The alpha value for blending style image and original image')
    parser.add_argument('--blend_beta', type=float, default=0.2, help='The beta value for fusing IP2P and MB')
    parser.add_argument('--image_guidance_scale', type=float, default=7.5, help='The image guidance scale')
    parser.add_argument('--dilate_strength', type=int, default=1, help='The dilate strength')
    parser.add_argument('--brightness', type=float, default=1, help='The brightness of the style image')

    args = parser.parse_args()
    return args
args = zone_parse_args()
sam_predictor = sam_model_registry["vit_h"](checkpoint=args.sam_ckpt_path,).to('cuda')  # 初始化sam
# choose the base model here
base_model_path = "data/ckpt/realisticVisionV60B1_v51VAE"
# base_model_path = "runwayml/stable-diffusion-v1-5"
# base_model_path = "CompVis/stable-diffusion-v1-4"  # 因为migc用的是sd1.4，和sd1.5的unet其实并不相同！
# brushnet用sd1.4也能跑起来。试试migc能不能用sd1.5跑起来？=>答案是可以的。所以都用sd1.5跑吧
# input brushnet ckpt path
# brushnet_path = "data/ckpt/segmentation_mask_brushnet_ckpt"
brushnet_path = "data/ckpt/random_mask_brushnet_ckpt"
# choose whether using blended operation
blended = False

# input source image / mask image path and the text prompt

# mask_path="src/mask_round.png"
# caption="A cake on the table."
# 原本的caption是str形式，这和migc的list[list[str]]差很多

# conditioning scale
brushnet_conditioning_scale=1.0

brushnet = BrushNetModel.from_pretrained(brushnet_path, torch_dtype=torch.float16)
pipe = StableDiffusionBrushNetPipeline.from_pretrained(
    base_model_path, brushnet=brushnet, torch_dtype=torch.float16, low_cpu_mem_usage=False
)

migc_ckpt_path = 'pretrained_weights/MIGC_SD14.ckpt'
assert os.path.isfile(migc_ckpt_path), "Please download the ckpt of migc and put it in the pretrained_weighrs/ folder!"
# pipe2 = StableDiffusionMIGCPipeline.from_pretrained(base_model_path)  # 都用sd1.5的权重 # 先确保两者的unet相同，才好将migc加进来
# 这里即使两者都用sd1.5进行from_pretrained，pipe.unet依旧不一样，令人困惑

pipe.attention_store = AttentionStore()  # migc部分
load_migc(pipe.unet, pipe.attention_store, migc_ckpt_path, attn_processor=MIGCProcessor)  # 这里处理了unet，使得需要额外参数作为输入
pipe = pipe.to("cuda")
# speed up diffusion process with faster scheduler and memory optimization
# pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# 下面这行是migc使用的scheduler，与上面一行存在矛盾。
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

# remove following line if xformers is not installed or when using Torch 2.0.
# pipe.enable_xformers_memory_efficient_attention()
# memory optimization.
pipe.enable_model_cpu_offload()

image_path="src/renlian.jpg"
init_image = cv2.imread(image_path)[:,:,::-1]
# mask_image = 1.*(cv2.imread(mask_path).sum(-1)>255)[:,:,np.newaxis]
mask_path1 = 'src/renlian_maozi_seg.jpg'
mask_path2 = 'src/renlian_glasses.jpg'
mask_path3 = 'src/renlian_scarf.jpg'
mask_path4 = 'src/renlian_medal.jpg'
# mask_path3 = 'src/renlian3_necklace.jpg'
# mask_path2 = 'src/sofa_mask2.jpg'
dilate_kernel = np.ones((3, 3), np.uint8)  # dilate核
dilate_iterations = 2
mask_image1, box_xyxy1, name1, d_mask_image1, mask_zone1 = read_mask(mask_path1, dilate_kernel, dilate_iterations)
mask_image2, box_xyxy2, name2, d_mask_image2, mask_zone2 = read_mask(mask_path2, dilate_kernel, dilate_iterations)
mask_image3, box_xyxy3, name3, d_mask_image3, mask_zone3 = read_mask(mask_path3, dilate_kernel, dilate_iterations)
mask_image4, box_xyxy4, name4, d_mask_image4, mask_zone4 = read_mask(mask_path4, dilate_kernel, dilate_iterations)
# cv2.imwrite('renlian_resize_mask1.png', mask_zone1)
# cv2.imwrite('renlian_resize_mask2.png', mask_zone2)
# cv2.imwrite('renlian_resize_mask3.png', mask_zone3)
# cv2.imwrite('renlian_resize_mask4.png', mask_zone4)
# 定义膨胀kernel  # 上面的mask_image都是(512, 512, 1)，所以给dilated_mask_image添加一维之后就可以直接相加了
mask_image_wo_d = mask_image1 + mask_image2 + mask_image3 + mask_image4# + mask_image3 + mask_image4  # 记录没有dilate的mask
mask_image = d_mask_image1 + d_mask_image2 + d_mask_image3 + d_mask_image4# + d_mask_image3 + d_mask_image4
mask_image[mask_image > 1.0] = 1.0  # 若mask有重叠，重叠区域相加会大于1，要把它们置为1
mask = mask_image  # 这里把mask_image给保存下来，方便后面画图
mask_wo_d = mask_image_wo_d
name_t = name1 + name2 + name3 + name4
name = 'output' + name_t + '.png'
###获取原图的h w###
h, w, _ = init_image.shape
h, w = cal_hw(h, w)
#################
init_image_c = init_image
mask_image_c = np.zeros_like(mask_image)  # mask_image中不需要编辑的区域对应为黑色
resize_image = cv2.resize(init_image_c, (w, h), interpolation=cv2.INTER_CUBIC)  # cv2.resize的img应为ndarray
resize_image = Image.fromarray(resize_image.astype(np.uint8)).convert("RGB")
resize_image.save('resize_renlian.png')
init_image_c = Image.fromarray(init_image_c.astype(np.uint8)).convert("RGB")
mask_image_c = Image.fromarray(mask_image_c.astype(np.uint8).repeat(3,-1)*255).convert("RGB")
init_image = init_image * (1-mask_image)
# init_image = init_image * mask_image


init_image = Image.fromarray(init_image.astype(np.uint8)).convert("RGB")

mask_image = Image.fromarray(mask_image.astype(np.uint8).repeat(3,-1)*255).convert("RGB")  # 最右边一维重复三遍，转黑白图再转RGB
# init_image.save('/home/xkzhu/yhx/BrushNet/examples/brushnet/init_image_T.png', quality=100)
# mask_image.save('/home/xkzhu/yhx/BrushNet/examples/brushnet/mask_renlian_seg.png', quality=100)
seed = 1234
seed_everything(seed)
original_image = cv2.imread('resize_renlian.png')


generator = torch.Generator("cuda").manual_seed(1234)
# 初始化migc需要的形参  # 需要把brushnet的prompt相关代码全部换成migc的
prompt_final = [['masterpiece, best quality, brown colored hat, black colored glasses, gray colored scarf, yellow colored medal',
                 'brown colored hat', 'black colored glasses', 'gray colored scarf', 'yellow colored medal'
                 ]]
bboxes = [[box_xyxy1, box_xyxy2, box_xyxy3, box_xyxy4]]  # 优化了代码，使其不再需要手动计算bboxes，更加智能！
########暂时注释掉，尝试zone的效果如何##########
# prompt_final = [['masterpiece, best quality, brown colored hat',
#                  'brown colored hat'
#                  ]]
# bboxes = [[box_xyxy1]]  # 优化了代码，使其不再需要手动计算bboxes，更加智能！
negative_prompt = 'worst quality, low quality, bad anatomy, watermark, text, blurry'

image = pipe(
    prompt_final,
    init_image,
    mask_image,
    num_inference_steps=50,
    generator=generator,
    brushnet_conditioning_scale=brushnet_conditioning_scale,
    # 加入migc相关的形参
    bboxes=bboxes,
    MIGCsteps=25,
    NaiveFuserSteps=50,
    aug_phase_with_and=False,
    negative_prompt=negative_prompt,
    # sa_preserve=True,  # sa_preserve和use_sa_preserve开启consistent-mig算法
    # use_sa_preserve=True,
    height=h,  # 这里的h w是否应该为16或者32的倍数？仅是8的倍数是不够的
    width=w   # 和预想的一样，仅为16的倍数也不够。要为32的倍数，且除以64的余数均同时为0或32才行=>不对，最好全为64的倍数
).images[0]
if blended:
    image_np=np.array(image)
    init_image_np=cv2.imread(image_path)[:,:,::-1]
    mask_np = 1.*(cv2.imread(mask_path).sum(-1)>255)[:,:,np.newaxis]

    # blur, you can adjust the parameters for better performance
    mask_blurred = cv2.GaussianBlur(mask_np*255, (21, 21), 0)/255
    mask_blurred = mask_blurred[:,:,np.newaxis]
    mask_np = 1-(1-mask_np) * (1-mask_blurred)

    image_pasted=init_image_np * (1-mask_np) + image_np*mask_np
    image_pasted=image_pasted.astype(image_np.dtype)
    image=Image.fromarray(image_pasted)
enhancer = ImageEnhance.Brightness(image)  # 使用图像增强看看效果
image = enhancer.enhance(args.brightness)
img1 = zone_ops(image, original_image, mask_zone1, args, sam_predictor)
cv2.imwrite('zone_multi_output1.png', img1)
img2 = zone_ops(image, img1, mask_zone2, args, sam_predictor)
cv2.imwrite('zone_multi_output2.png', img2)
img3 = zone_ops(image, img2, mask_zone3, args, sam_predictor)
cv2.imwrite('zone_multi_output3.png', img3)
img4 = zone_ops(image, img3, mask_zone4, args, sam_predictor)
cv2.imwrite('zone_multi_output4.png', img4)
image.save(name)
image = pipe.draw_box_desc(image, bboxes[0], prompt_final[0][1:])
mask_img = pipe.draw_mask(image, mask)
mask_img = pipe.draw_mask(mask_img, mask_wo_d)
mask_img = Image.fromarray(mask_img.astype(np.uint8)).convert('RGB')
image.save('anno_output.png')
mask_img.save('anno_mask.png')
