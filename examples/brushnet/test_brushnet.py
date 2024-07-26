import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 本地改镜像站才能调试
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
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

def read_mask(mask_path, kernel):
    name = mask_path[4: -4]
    mask_image = 1.*(cv2.imread(mask_path).sum(-1)>255)[:,:,np.newaxis]
    dilated_mask_image = cv2.dilate(mask_image, kernel, iterations=1)[..., np.newaxis]
    mask_image_t = torch.from_numpy(mask_image).permute(2, 0, 1)
    box_xyxy = masks_to_boxes(mask_image_t)
    h, w, _ = mask_image.shape
    box_xyxy = box_xyxy.squeeze(0)
    box_xyxy[0], box_xyxy[2] = box_xyxy[0] / w, box_xyxy[2] / w
    box_xyxy[1], box_xyxy[3] = box_xyxy[1] / h, box_xyxy[3] / h
    return mask_image, box_xyxy.numpy().tolist(), name, dilated_mask_image
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


# choose the base model here
base_model_path = "data/ckpt/realisticVisionV60B1_v51VAE"
# base_model_path = "runwayml/stable-diffusion-v1-5"
# base_model_path = "Uminosachi/realisticVisionV51_v51VAE-inpainting"
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

image_path="src/test_image.jpg"
init_image = cv2.imread(image_path)[:,:,::-1]
# mask_image = 1.*(cv2.imread(mask_path).sum(-1)>255)[:,:,np.newaxis]
mask_path1 = 'src/mask_round1.png'
mask_path2 = 'src/mask_round2.png'
# mask_path3 = 'src/renlian3_necklace.jpg'
# mask_path2 = 'src/sofa_mask2.jpg'
kernel = np.ones((15, 15), np.uint8)  # dilate核
mask_image1, box_xyxy1, name1, d_mask_image1 = read_mask(mask_path1, kernel)
mask_image2, box_xyxy2, name2, d_mask_image2 = read_mask(mask_path2, kernel)
# mask_image3, box_xyxy3, name3, d_mask_image3 = read_mask(mask_path3, kernel)
# mask_image4, box_xyxy4, name4, d_mask_image4 = read_mask(mask_path4, kernel)
# 定义膨胀kernel  # 上面的mask_image都是(512, 512, 1)，所以给dilated_mask_image添加一维之后就可以直接相加了
mask_image_wo_d = mask_image1 + mask_image2# + mask_image3 + mask_image4# + mask_image3 + mask_image4  # 记录没有dilate的mask
mask_image = mask_image1 + mask_image2# + d_mask_image3 + d_mask_image4# + d_mask_image3 + d_mask_image4
mask_image[mask_image > 1.0] = 1.0  # 若mask有重叠，重叠区域相加会大于1，要把它们置为1
mask = mask_image  # 这里把mask_image给保存下来，方便后面画图
mask_wo_d = mask_image_wo_d
name_t = name1 + name2# + name3 + name4
name = 'output' + name_t + '.png'

init_image_c = init_image
mask_image_c = np.zeros_like(mask_image)
init_image_c = Image.fromarray(init_image_c.astype(np.uint8)).convert("RGB")
mask_image_c = Image.fromarray(mask_image_c.astype(np.uint8).repeat(3,-1)*255).convert("RGB")
init_image = init_image * (1-mask_image)
# init_image = init_image * mask_image
h, w, _ = init_image.shape
h, w = cal_hw(h, w)
init_image = Image.fromarray(init_image.astype(np.uint8)).convert("RGB")
mask_image = Image.fromarray(mask_image.astype(np.uint8).repeat(3,-1)*255).convert("RGB")  # 最右边一维重复三遍，转黑白图再转RGB
# init_image.save('/home/xkzhu/yhx/BrushNet/examples/brushnet/init_image_T.png', quality=100)
# mask_image.save('/home/xkzhu/yhx/BrushNet/examples/brushnet/mask_renlian_seg.png', quality=100)
generator = torch.Generator("cuda").manual_seed(1234)
seed = 1234
seed_everything(seed)
# 初始化migc需要的形参  # 需要把brushnet的prompt相关代码全部换成migc的
# prompt_final = [['masterpiece, best quality, orange colored orange, yellow colored lemon',
#                  'orange colored orange', 'yellow colored lemon']]  # 用migc的multi instances prompt才能实现多物体控制
prompt_final = [['']]  # 什么都没有，啥也不编辑跑一次  # 判断prompt_final是否为0以关闭brushnet的branch，这使得模型能够输出一模一样的图片
bboxes = [[]]  # 优化了代码，使其不再需要手动计算bboxes，更加智能！
negative_prompt = 'worst quality, low quality, bad anatomy, watermark, text, blurry'

image = pipe(
    prompt_final,
    init_image_c,
    mask_image_c,
    num_inference_steps=50,
    generator=generator,
    brushnet_conditioning_scale=brushnet_conditioning_scale,
    # 加入migc相关的形参
    bboxes=bboxes,
    MIGCsteps=25,
    NaiveFuserSteps=25,
    BaSteps=0,
    aug_phase_with_and=False,
    negative_prompt=negative_prompt,
    sa_preserve=True,  # sa_preserve和use_sa_preserve开启consistent-mig算法
    # use_sa_preserve=True,
    height=h,  # 这里的h w是否应该为16或者32的倍数？仅是8的倍数是不够的
    width=w   # 和预想的一样，仅为16的倍数也不够。要为32的倍数，且除以64的余数均同时为0或32才行=>不对，最好全为64的倍数
).images[0]

image.save('before.png')

prompt_final = [['masterpiece, best quality, orange colored orange, yellow colored lemon',
                 'orange colored orange', 'yellow colored lemon'
                 ]]
bboxes = [[box_xyxy1, box_xyxy2]]  # 优化了代码，使其不再需要手动计算bboxes，更加智能！


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
    NaiveFuserSteps=25,
    BaSteps=25,
    aug_phase_with_and=False,
    negative_prompt=negative_prompt,
    sa_preserve=True,  # sa_preserve和use_sa_preserve开启consistent-mig算法
    use_sa_preserve=True,
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

image.save(name)
image = pipe.draw_box_desc(image, bboxes[0], prompt_final[0][1:])
mask_img = pipe.draw_mask(image, mask)
mask_img = pipe.draw_mask(mask_img, mask_wo_d)
mask_img = Image.fromarray(mask_img.astype(np.uint8)).convert('RGB')
image.save('anno_output.png')
mask_img.save('anno_mask.png')
