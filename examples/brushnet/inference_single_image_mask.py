import yaml
from diffusers import EulerDiscreteScheduler
from migc.migc_utils import seed_everything
from migc.migc_pipeline import StableDiffusionMIGCPipeline, MIGCProcessor, AttentionStore
import os
import numpy
from PIL import Image

def get_info_from_image(path, height, width):
    mask = Image.open(path)
    mask = numpy.array(mask)
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

if __name__ == '__main__':
    migc_ckpt_path = 'pretrained_weights/MIGC_plus_SD14.ckpt'
    assert os.path.isfile(migc_ckpt_path), "Please download the ckpt of migc and put it in the pretrained_weighrs/ folder!"

    sd1x_path = '/mnt/data1/dewei/ckpt/stable-diffusion-v1-4' if os.path.isdir('/mnt/data1/dewei/ckpt/stable-diffusion-v1-4') else "CompVis/stable-diffusion-v1-4"
    # MIGC is a plug-and-play controller.
    # You can go to https://civitai.com/search/models?baseModel=SD%201.4&baseModel=SD%201.5&sortBy=models_v5 find a base model with better generation ability to achieve better creations.
    
    # Construct MIGC pipeline
    pipe = StableDiffusionMIGCPipeline.from_pretrained(
        sd1x_path)
    pipe.attention_store = AttentionStore()
    from migc.migc_utils import load_migc_plus
    load_migc_plus(pipe.unet , pipe.attention_store,
            migc_ckpt_path, attn_processor=MIGCProcessor)
    pipe = pipe.to("cuda")
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

    prompt_final = [['masterpiece, best quality, a grassland with sunflower, blue sky, black horse, a white house']]
    bboxes = [[]]
    masks = [[]]
    
    from glob import glob
    mask_dir = './mask_example'
    for mask_path in glob(mask_dir + '/*.png'):
        bbox, mask = get_info_from_image(mask_path, 512, 512)
        bboxes[0].append(bbox)
        masks[0].append(mask)
        prompt_final[0].append(mask_path.split('/')[-1].split('.')[0])
    
    negative_prompt = 'worst quality, low quality, bad anatomy, watermark, text, blurry'
    for seed in range(10):
        seed_everything(seed)
        image = pipe(prompt_final, bboxes, masks, num_inference_steps=50, guidance_scale=7.5, 
                        MIGCsteps=25, NaiveFuserSteps=50, aug_phase_with_and=False, negative_prompt=negative_prompt).images[0]
        image.save(f'output_{seed}.png')
        # image.show()
        image = pipe.draw_box_desc(image, bboxes[0], prompt_final[0][1:])
        image.save(f'anno_output_{seed}.png')
        # image.show()