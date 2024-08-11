import argparse
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 本地改镜像站才能调试
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import random
import sys
import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageFilter
from safetensors.torch import load_model
from transformers import CLIPTextModel, DPTFeatureExtractor, DPTForDepthEstimation

from diffusers import UniPCMultistepScheduler, EulerDiscreteScheduler
from diffusers.pipelines.controlnet.pipeline_controlnet import ControlNetModel
from powerpaint.models.BrushNet_CA import BrushNetModel
from powerpaint.models.unet_2d_condition import UNet2DConditionModel
from powerpaint.pipelines.pipeline_PowerPaint import StableDiffusionInpaintPipeline as Pipeline
from powerpaint.pipelines.pipeline_PowerPaint_Brushnet_CA import StableDiffusionPowerPaintBrushNetPipeline
from powerpaint.pipelines.pipeline_PowerPaint_ControlNet import (
    StableDiffusionControlNetInpaintPipeline as controlnetPipeline,
)
from powerpaint.utils.utils import TokenizerWrapper, add_tokens
###########下面为新加的import###########
from torchvision.ops import masks_to_boxes
from migc_plus.migc_utils import seed_everything
from migc_plus.migc_pipeline import StableDiffusionMIGCPipeline, MIGCProcessor, AttentionStore
from migc_plus.migc_utils import load_migc
from mask_utils import *
from PIL import ImageEnhance
###########下面加入PCT_Net相关#############
# sys.path.insert(0, '.')
from iharm.inference.predictor import Predictor
from iharm.inference.utils import load_model_pct
from iharm.mconfigs import ALL_MCONFIGS

torch.set_grad_enabled(False)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def add_task(prompt, negative_prompt, control_type, version):
    pos_prefix = neg_prefix = ""
    if control_type == "object-removal" or control_type == "image-outpainting":
        promptA = pos_prefix + " P_ctxt"
        promptB = pos_prefix + " P_ctxt"
        negative_promptA = neg_prefix + " P_obj"
        negative_promptB = neg_prefix + " P_obj"
    elif control_type == "shape-guided":
        promptA = pos_prefix + " P_shape"
        promptB = pos_prefix + " P_ctxt"
        negative_promptA = neg_prefix + "P_shape"
        negative_promptB = neg_prefix + "P_ctxt"
    else:
        promptA = pos_prefix + " P_obj"
        promptB = pos_prefix + " P_obj"
        negative_promptA = neg_prefix + "P_obj"
        negative_promptB = neg_prefix + "P_obj"

    return promptA, promptB, negative_promptA, negative_promptB


def select_tab_text_guided():
    return "text-guided"


def select_tab_object_removal():
    return "object-removal"


def select_tab_image_outpainting():
    return "image-outpainting"


def select_tab_shape_guided():
    return "shape-guided"


def get_info_from_image(path, height, width):
    mask = Image.open(path)
    mask = np.array(mask)
    assert mask.shape[0] == height and mask.shape[1] == width, "The mask should have the same size as the generated image!"
    mask = mask // 255
    if len(mask.shape) == 3:
        mask = mask[: ,: ,0]
    bbox = [1.0, 1.0, 0.0, 0.0]
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] == 1:
                bbox[0] = min(bbox[0], j / width)
                bbox[1] = min(bbox[1], i / height)
                bbox[2] = max(bbox[2], j / width)
                bbox[3] = max(bbox[3], i / height)
    return bbox, mask

class PowerPaintController:
    def __init__(self, weight_dtype, checkpoint_dir, local_files_only, version) -> None:
        self.version = version
        self.checkpoint_dir = checkpoint_dir
        self.local_files_only = local_files_only

        # initialize powerpaint pipeline
        # brushnet-based version
        unet = UNet2DConditionModel.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            subfolder="unet",
            revision=None,
            torch_dtype=weight_dtype,
            local_files_only=local_files_only,
        )
        text_encoder_brushnet = CLIPTextModel.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            subfolder="text_encoder",
            revision=None,
            torch_dtype=weight_dtype,
            local_files_only=local_files_only,
        )
        brushnet = BrushNetModel.from_unet(unet)
        base_model_path = os.path.join(checkpoint_dir, "realisticVisionV60B1_v51VAE")
        self.pipe = StableDiffusionPowerPaintBrushNetPipeline.from_pretrained(
            base_model_path,
            brushnet=brushnet,
            text_encoder_brushnet=text_encoder_brushnet,
            torch_dtype=weight_dtype,
            low_cpu_mem_usage=False,
            safety_checker=None,
        )
        self.pipe.unet = UNet2DConditionModel.from_pretrained(
            base_model_path,
            subfolder="unet",
            revision=None,
            torch_dtype=weight_dtype,
            local_files_only=local_files_only,
        )
        self.pipe.tokenizer = TokenizerWrapper(
            from_pretrained=base_model_path,
            subfolder="tokenizer",
            revision=None,
            torch_type=weight_dtype,
            local_files_only=local_files_only,
        )

        # add learned task tokens into the tokenizer
        add_tokens(
            tokenizer=self.pipe.tokenizer,
            text_encoder=self.pipe.text_encoder_brushnet,
            placeholder_tokens=["P_ctxt", "P_shape", "P_obj"],
            initialize_tokens=["a", "a", "a"],
            num_vectors_per_token=10,
        )
        load_model(
            self.pipe.brushnet,
            os.path.join(checkpoint_dir, "PowerPaint_Brushnet/diffusion_pytorch_model.safetensors"),
        )

        self.pipe.text_encoder_brushnet.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "PowerPaint_Brushnet/pytorch_model.bin")), strict=False
        )
        migc_ckpt_path = 'pretrained_weights/MIGC_plus_SD14.ckpt'
        self.pipe.attention_store = AttentionStore()  # migc部分
        from migc_plus.migc_utils import load_migc_plus  # 引入migc++
        load_migc_plus(self.pipe.unet, self.pipe.attention_store,
                       migc_ckpt_path, attn_processor=MIGCProcessor)  # 这里处理了unet，使得需要额外参数作为输入
        # self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)  # 应该换成migc++的EulerDiscreteScheduler吗？
        self.pipe.scheduler = EulerDiscreteScheduler.from_config(self.pipe.scheduler.config)  # 这个都可以试试的
        self.pipe.enable_model_cpu_offload()
        self.pipe = self.pipe.to("cuda")

    def predict(
            self,
            input_image=None,
            prompt=None,
            fitting_degree=None,
            ddim_steps=None,
            scale=None,
            seed=None,
            negative_prompt=None,
            task=None,
            vertical_expansion_ratio=None,
            horizontal_expansion_ratio=None,
            # 加入migc相关的形参
            bboxes=None,
            masks=None,
            MIGCsteps=25,
            NaiveFuserSteps=50,
            aug_phase_with_and=False,
            # sa_preserve=True,  # sa_preserve和use_sa_preserve开启consistent-mig算法
            # use_sa_preserve=True,
            h=None,  # 这里的h w是否应该为16或者32的倍数？仅是8的倍数是不够的
            w=None  # 和预想的一样，仅为16的倍数也不够。要为32的倍数，且除以64的余数均同时为0或32才行=>不对，最好全为64的倍数
    ):
        # size1, size2 = input_image["image"].convert("RGB").size

        # input_image["image"] = cv2.resize(input_image["image"], (w, h), interpolation=cv2.INTER_CUBIC)
        # input_image["mask"] = cv2.resize(input_image["mask"], (w, h), interpolation=cv2.INTER_CUBIC)
        # 这个resize函数换成app_brushnet的

        ###########暂时不需要vertical和horizontal的expansion##########

        if self.version != "ppt-v1":
            if task == "image-outpainting":
                prompt = prompt + " empty scene"
            if task == "object-removal":
                prompt = prompt + " empty scene blur"
        promptA, promptB, negative_promptA, negative_promptB = add_task(prompt, negative_prompt, task, self.version)
        print(promptA, promptB, negative_promptA, negative_promptB)

        seed_everything(seed)  # 把原本的set_seed改成seed_everything

        # for brushnet-based method
        np_inpimg = np.array(input_image["image"])
        np_inmask = np.array(input_image["mask"]) / 255.0  # 这里直接传nparray
        np_inpimg = np_inpimg * (1 - np_inmask)
        input_image["image"] = Image.fromarray(np_inpimg.astype(np.uint8)).convert(
            "RGB")  # 这儿的image对应brushnet的masked_image
        result = self.pipe(
            promptA=promptA,
            promptB=promptB,
            promptU=prompt,  # prompt和negative_prompt就用migc的版本。其他两个看了代码发现根本不用管
            tradoff=fitting_degree,  # fitting_degree每个物体应该要不一样的，看看之后怎么修改吧
            tradoff_nag=fitting_degree,
            image=input_image["image"].convert("RGB"),
            mask=input_image["mask"].convert("RGB"),
            num_inference_steps=ddim_steps,  # 这儿就是50
            generator=torch.Generator("cuda").manual_seed(seed),
            brushnet_conditioning_scale=1.0,
            negative_promptA=negative_promptA,
            negative_promptB=negative_promptB,
            negative_promptU=negative_prompt,
            guidance_scale=scale,
            bboxes=bboxes,
            masks=masks,
            MIGCsteps=MIGCsteps,
            NaiveFuserSteps=NaiveFuserSteps,
            aug_phase_with_and=aug_phase_with_and,
            width=w,
            height=h,
        ).images[0]

        return result

    def infer(
            self,
            input_image=None,
            text_guided_prompt=None,
            text_guided_negative_prompt=None,
            shape_guided_prompt=None,
            shape_guided_negative_prompt=None,
            fitting_degree=None,
            ddim_steps=50,
            scale=7.5,
            seed=None,
            task=None,
            vertical_expansion_ratio=None,
            horizontal_expansion_ratio=None,
            outpaint_prompt=None,
            outpaint_negative_prompt=None,
            removal_prompt=None,
            removal_negative_prompt=None,
            enable_control=False,
            input_control_image=None,
            control_type="canny",
            controlnet_conditioning_scale=None,
            # 加入migc相关的形参
            bboxes=None,
            masks=None,
            MIGCsteps=25,
            NaiveFuserSteps=50,
            aug_phase_with_and=False,
            # sa_preserve=True,  # sa_preserve和use_sa_preserve开启consistent-mig算法
            # use_sa_preserve=True,
            height=None,  # 这里的h w是否应该为16或者32的倍数？仅是8的倍数是不够的
            width=None  # 和预想的一样，仅为16的倍数也不够。要为32的倍数，且除以64的余数均同时为0或32才行=>不对，最好全为64的倍数
    ):
        if task == "text-guided":
            prompt = text_guided_prompt
            negative_prompt = text_guided_negative_prompt
        elif task == "shape-guided":
            prompt = shape_guided_prompt
            negative_prompt = shape_guided_negative_prompt
        elif task == "object-removal":
            prompt = removal_prompt
            negative_prompt = removal_negative_prompt
        elif task == "image-outpainting":
            prompt = outpaint_prompt
            negative_prompt = outpaint_negative_prompt
            return self.predict(
                input_image,
                prompt,
                fitting_degree,
                ddim_steps,
                scale,
                seed,
                negative_prompt,
                task,
                vertical_expansion_ratio,
                horizontal_expansion_ratio,
            )
        else:
            task = "text-guided"
            prompt = text_guided_prompt
            negative_prompt = text_guided_negative_prompt

        # currently, we only support controlnet in PowerPaint-v1
        image = self.predict(
            input_image=input_image,
            prompt=prompt,
            fitting_degree=fitting_degree,
            ddim_steps=ddim_steps,
            scale=scale,
            seed=seed,
            negative_prompt=negative_prompt,
            task=task,
            vertical_expansion_ratio=None,
            horizontal_expansion_ratio=None,
            bboxes=bboxes,
            masks=masks,
            MIGCsteps=MIGCsteps,
            NaiveFuserSteps=NaiveFuserSteps,
            aug_phase_with_and=aug_phase_with_and,
            h=height,
            w=width,
        )

        return image



def read_mask(mask_path, kernel, iterations):
    name = mask_path[4: -4]
    mask = cv2.imread(mask_path)
    mask_image = 1. * (cv2.imread(mask_path).sum(-1) > 255)[:, :, np.newaxis]
    dilated_mask_image = cv2.dilate(mask_image, kernel, iterations)[..., np.newaxis]
    mask_image_t = torch.from_numpy(mask_image).permute(2, 0, 1)
    box_xyxy = masks_to_boxes(mask_image_t)
    h, w, _ = mask_image.shape
    box_xyxy = box_xyxy.squeeze(0)
    box_xyxy[0], box_xyxy[2] = box_xyxy[0] / w, box_xyxy[2] / w
    box_xyxy[1], box_xyxy[3] = box_xyxy[1] / h, box_xyxy[3] / h
    return mask_image, box_xyxy.numpy().tolist(), name, dilated_mask_image, mask
def zone_ops(style_image, original_image, mask):
    """

    Args:
        style_image: sd output image
        original_image: resized image or edited zone_ops image
        mask: input mask
        args: parser's args

    Returns: edited zone_ops image

    """
    original_image_np = np.array(original_image)
    original_image_np = cv2.cvtColor(original_image_np, cv2.COLOR_BGR2RGB)

    _, mask_output = get_finemask_everything(mask, style_image, sam_predictor)
    _, mask_init = get_finemask_everything(mask, original_image, sam_predictor, mask_output)
      # masked_out是被mask后的style_image，mask是优化后的Mask
    kernel = np.ones((3, 3), np.uint8)
    mask_init = cv2.dilate(mask_init, kernel=kernel, iterations=3)
    mask_pct = mask_init - mask_output
    mask_pct[mask_pct < 255] = 0
    mask_pct = mask_pct / 255
    mask_init_blurred = cv2.GaussianBlur(mask_init, (11, 11), 0)
    mask_init_blurred = mask_init_blurred[..., np.newaxis] / 255

    # dilated_mask_all, _ = dilate_mask(mask_image, iterations=args.dilate_strength)  # 确实需要dilate，因为物体不一定能完全和想要的区域吻合

    style_image_np = cv2.cvtColor(np.array(style_image), cv2.COLOR_BGR2RGB)
    #####进行替换#####
    # cv2.imwrite('10.png', original_image_np * (1 - mask_init_blurred))
    # cv2.imwrite('11.png', style_image_np * mask_init_blurred)
    original_image = original_image_np * (1 - mask_init_blurred) + style_image_np * mask_init_blurred  # 到这里都没有问题
    # mask_init_lr = cv2.resize(mask_init_pct, (256, 256))
    # original_image_lr = cv2.resize(original_image, (256, 256))
    # _, original_image = pct_predictor.predict(original_image_lr, original_image, mask_init_lr, mask_init_pct)
    gt_image_np = original_image  # original_image由于是直接imread的，所以不用反色

    mask_blurred = cv2.GaussianBlur(mask_output, (11, 11), 0)
    mask_blurred = mask_blurred[..., np.newaxis] / 255
    new_canvas = gt_image_np * (1 - mask_blurred) + style_image_np * mask_blurred
    new_canvas_lr = cv2.resize(new_canvas, (256, 256))
    mask_lr = cv2.resize(mask_pct, (256, 256))
    _, new_canvas = pct_predictor.predict(new_canvas_lr, new_canvas, mask_lr, mask_pct)
    new_canvas = cv2.cvtColor(new_canvas.astype(np.uint8), cv2.COLOR_BGR2RGB)
    new_canvas = Image.fromarray(new_canvas).convert("RGB")
    return new_canvas
if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--weight_dtype", type=str, default="float16")  # 原本是float16，我就要改为float32
    args.add_argument("--checkpoint_dir", type=str, default="./data/ckpt")
    args.add_argument("--version", type=str, default="ppt-v2")
    args.add_argument("--share", action="store_true")
    args.add_argument(
        "--local_files_only", action="store_true", help="enable it to use cached files without requesting from the hub"
    )
    args.add_argument('--sam_ckpt_path', type=str, default="./data/ckpt/sam_vit_h_4b8939.pth",
                        help='The path to SAM checkpoint')
    args.add_argument('--threshold', type=int, default=20, help='The threshold of edge smoother')
    args.add_argument('--alpha', type=float, default=0.3,
                        help='The alpha value for blending style image and original image')
    args.add_argument('--blend_beta', type=float, default=0.2, help='The beta value for fusing IP2P and MB')
    args.add_argument('--image_guidance_scale', type=float, default=7.5, help='The image guidance scale')
    args.add_argument('--dilate_strength', type=int, default=1, help='The dilate strength')
    args.add_argument('--brightness', type=float, default=1, help='The brightness of the style image')
    args.add_argument('--gpu', type=str, default='cuda', help='ID of used GPU.')
    args.add_argument('--model_type', default='ViT_pct')
    args.add_argument('--weights', default='pretrained_weights/PCTNet_ViT.pth', help='path to the pctnet weights')
    args = args.parse_args()
    # initialize the sam_predictor
    sam_predictor = sam_model_registry["vit_h"](checkpoint=args.sam_ckpt_path, ).to('cuda')  # 初始化sam
    # initialize the pct_net
    # Load model
    pct_model = load_model_pct(args.model_type, args.weights, verbose=False)

    device = torch.device(args.gpu)
    use_attn = ALL_MCONFIGS[args.model_type]['params']['use_attn']
    normalization = ALL_MCONFIGS[args.model_type]['params']['input_normalization']
    pct_predictor = Predictor(pct_model, device, use_attn=use_attn, mean=normalization['mean'], std=normalization['std'])
    # initialize the pipeline controller
    weight_dtype = torch.float16 if args.weight_dtype == "float16" else torch.float32

    controller = PowerPaintController(weight_dtype, args.checkpoint_dir, args.local_files_only,
                                      args.version)  # 这里实例化了PowerPaintController

    # image = input_image["image"].convert("RGB"),
    # mask = input_image["mask"].convert("RGB"),
    # 总之input_image是个dict，包含了image和mask
    image_path = "src/renxiang14.jpg"
    init_image = cv2.imread(image_path)[:, :, ::-1]
    h, w, _ = init_image.shape  # 从现在开始，所有的init_image都是被resize之后再传进来的
    # h, w = cal_hw(h, w)
    mask_path1 = 'src/renxiang14_square_glove1.jpg'
    mask_path2 = 'src/renxiang14_square_glove2.jpg'
    mask_path3 = 'src/renxiang14_glove_2.jpg'
    mask_path4 = 'src/renlian_medal.jpg'
    # Shape-guided object inpainting
    prompt_final = [['masterpiece, best quality, pink colored glove, green colored glove']]
    negative_prompt = 'worst quality, low quality, bad anatomy, watermark, text, blurry'
    dilate_kernel = np.ones((3, 3), np.uint8)  # dilate核
    dilate_iterations = 2
    mask_image1, box_xyxy1, name1, d_mask_image1, mask_zone1 = read_mask(mask_path1, dilate_kernel, dilate_iterations)
    mask_image2, box_xyxy2, name2, d_mask_image2, mask_zone2 = read_mask(mask_path2, dilate_kernel, dilate_iterations)
    # mask_image3, box_xyxy3, name3, d_mask_image3, mask_zone3 = read_mask(mask_path3, dilate_kernel, dilate_iterations)
    # mask_image4, box_xyxy4, name4, d_mask_image4, mask_zone4 = read_mask(mask_path4, dilate_kernel, dilate_iterations)
    mask_image_wo_d = mask_image1 + mask_image2# + mask_image3# + mask_image4  # 记录没有dilate的mask
    mask_image = mask_image1 + mask_image2# + mask_image3# + mask_image4  # 先试一下不膨胀的
    mask_image[mask_image > 1.0] = 1.0  # 若mask有重叠，重叠区域相加会大于1，要把它们置为1
    name = name1 + name2# + name3# + name4
    name = 'output_' + name + '.png'
    ##############################
    init_image_zone = Image.open(image_path)
    # init_image_zone.save('101.png')
    init_image = Image.fromarray(init_image.astype(np.uint8)).convert("RGB")  # brushnet的输入需要为Image形式
    mask_image = Image.fromarray(mask_image.astype(np.uint8).repeat(3, -1) * 255).convert("RGB")
    input_image = {}
    input_image["image"] = init_image
    input_image["mask"] = mask_image
    bboxes = [[]]  # 优化了代码，使其不再需要手动计算bboxes，更加智能！
    masks = [[]]

    bbox, mask = get_info_from_image(mask_path1, h, w)
    bboxes[0].append(bbox)
    masks[0].append(mask)
    prompt_final[0].append('pink colored glove')
    bbox, mask = get_info_from_image(mask_path2, h, w)
    bboxes[0].append(bbox)
    masks[0].append(mask)
    prompt_final[0].append('green colored glove')
    # bbox, mask = get_info_from_image(mask_path3, h, w)
    # bboxes[0].append(bbox)
    # masks[0].append(mask)
    # prompt_final[0].append('yellow colored glove')
    # bbox, mask = get_info_from_image(mask_path4, h, w)
    # bboxes[0].append(bbox)
    # masks[0].append(mask)
    # prompt_final[0].append('yellow colored medal')
    # fitting_degree = 0.55  # 不太好说每个object能不能享有不同的fitting_degree，由于只有一张mask_image作为输入，model应该没有这么智能
    ########      参数传递流程：infer->predict->pipe     ########
    ##从pipe由下往上解决，肯定不会错
    image = controller.infer(
        input_image,
        text_guided_prompt=prompt_final,
        text_guided_negative_prompt=negative_prompt,
        shape_guided_prompt=prompt_final,
        shape_guided_negative_prompt=negative_prompt,
        fitting_degree=0.5,
        ddim_steps=50,
        scale=7.5,
        seed=1234,
        task="shape-guided",
        # 加入migc相关的形参
        bboxes=bboxes,
        masks=masks,
        MIGCsteps=25,
        NaiveFuserSteps=50,
        aug_phase_with_and=False,
        # sa_preserve=True,  # sa_preserve和use_sa_preserve开启consistent-mig算法
        # use_sa_preserve=True,
        height=h,  # 这里的h w是否应该为16或者32的倍数？仅是8的倍数是不够的
        width=w  # 和预想的一样，仅为16的倍数也不够。要为32的倍数，且除以64的余数均同时为0或32才行=>不对，最好全为64的倍数
    )
    # path_a = "output_renxiang6_hatrenxiang6_blouse.png"
    # image = Image.open(path_a) # 读出来本来就是RGB，所以不需要再convert("RGB")了
    image.save(name)
    # enhancer = ImageEnhance.Brightness(image)  # 对生成结果使用亮度增强，增强幅度为1.1
    # image = enhancer.enhance(args.brightness)
    # image.save('bright' + name)
    img1 = zone_ops(image, init_image_zone, mask_zone1)
    img1.save('ppt_zone_multi_output1.png')
    img2 = zone_ops(image, img1, mask_zone2)
    img2.save('ppt_zone_multi_output2.png')
    # img3 = zone_ops(image, img2, mask_zone3)
    # img3.save('ppt_zone_multi_output3.png')
    # img4 = zone_ops(image, img3, mask_zone4)
    # img4.save('ppt_zone_multi_output4.png', img4)

