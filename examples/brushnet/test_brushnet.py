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

# choose the base model here
# base_model_path = "data/ckpt/realisticVisionV60B1_v51VAE"
base_model_path = "runwayml/stable-diffusion-v1-5"
# base_model_path = "CompVis/stable-diffusion-v1-4"  # 因为migc用的是sd1.4，和sd1.5的unet其实并不相同！
# brushnet用sd1.4也能跑起来。试试migc能不能用sd1.5跑起来？=>答案是可以的。所以都用sd1.5跑吧
# input brushnet ckpt path
brushnet_path = "data/ckpt/segmentation_mask_brushnet_ckpt"

# choose whether using blended operation
blended = False

# input source image / mask image path and the text prompt
image_path="src/test_image.jpg"
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

init_image = cv2.imread(image_path)[:,:,::-1]
# mask_image = 1.*(cv2.imread(mask_path).sum(-1)>255)[:,:,np.newaxis]
mask_path1 = 'src/mask_round1.png'
mask_path2 = 'src/mask_round2.png'
mask_image1 = 1.*(cv2.imread(mask_path1).sum(-1)>255)[:,:,np.newaxis]
mask_image_t1 = torch.from_numpy(mask_image1).permute(2, 0, 1)
mask_image2 = 1.*(cv2.imread(mask_path2).sum(-1)>255)[:,:,np.newaxis]
mask_image_t2 = torch.from_numpy(mask_image2).permute(2, 0, 1)
mask_image = mask_image1 + mask_image2

# mask3.save('/home/xkzhu/yhx/BrushNet/examples/brushnet/mask_round12.png', quality=100)
boxes_xyxy = []
boxes_xyxy.append(masks_to_boxes(mask_image_t1))
boxes_xyxy.append(masks_to_boxes(mask_image_t2))
# boxes_xyxy.save('/home/xkzhu/yhx/BrushNet/examples/brushnet/mask2box.png', quality=100)
init_image = init_image * (1-mask_image)
# init_image = init_image * mask_image

init_image = Image.fromarray(init_image.astype(np.uint8)).convert("RGB")
mask_image = Image.fromarray(mask_image.astype(np.uint8).repeat(3,-1)*255).convert("RGB")
init_image.save('/home/xkzhu/yhx/BrushNet/examples/brushnet/init_image_T.png', quality=100)
mask_image.save('/home/xkzhu/yhx/BrushNet/examples/brushnet/mask_round12.png', quality=100)
generator = torch.Generator("cuda").manual_seed(1234)
# 初始化migc需要的形参  # 需要把brushnet的prompt相关代码全部换成migc的
# prompt_final = [['masterpiece, best quality, red colored apple, purple colored ball, yellow colored banana, green colored watermelon',
#                  'red colored apple', 'purple colored ball', 'yellow colored banana', 'green colored watermelon']]  # 用migc的multi instances prompt才能实现多物体控制
# bboxes = [[[0.1, 0.1, 0.3, 0.3], [0.7, 0.1, 0.9, 0.3], [0.1, 0.7, 0.3, 0.9], [0.7, 0.7, 0.9, 0.9]]]
prompt_final = [['masterpiece, best quality, orange colored orange, yellow colored lemon',
                 'orange colored orange', 'yellow colored lemon']]  # 用migc的multi instances prompt才能实现多物体控制
bboxes = [[[0.2871, 0.1992, 0.4746, 0.373], [0.5742, 0.2558, 0.748, 0.42578]]]
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
    aug_phase_with_and=False,
    negative_prompt=negative_prompt
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

image.save("output.png")