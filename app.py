import gradio as gr

from diffusers import DiffusionPipeline
import torch

import base64
from io import BytesIO
import os
import gc
import random
import time
from PIL import PngImagePlugin
from datetime import datetime

# Only used when MULTI_GPU set to True
from helper import UNetDataParallel
from share_btn import community_icon_html, loading_icon_html, share_js

# SDXL code: https://github.com/huggingface/diffusers/pull/3859

model_dir = os.getenv("SDXL_MODEL_DIR")

if model_dir:
    # Use local model
    model_key_base = os.path.join(model_dir, "LCM_Dreamshaper_v7")
else:
    model_key_base = "SimianLuo/LCM_Dreamshaper_v7"

# Process environment variables

# Use refiner (enabled by default)
enable_refiner = False
# Output images before the refiner and after the refiner
output_images_before_refiner = True

# Generate how many images by default
default_num_images = int(os.getenv("DEFAULT_NUM_IMAGES", "4"))
if default_num_images < 1:
    default_num_images = 1

# Create public link
share = os.getenv("SHARE", "false").lower() == "true"

print("Loading model", model_key_base)
pipe = DiffusionPipeline.from_pretrained(model_key_base, custom_pipeline="latent_consistency_txt2img",
                                         custom_revision="main", from_flax=False, safety_checker=None)
pipe.to(torch_device="cuda", torch_dtype=torch.float16)

multi_gpu = os.getenv("MULTI_GPU", "false").lower() == "true"

if multi_gpu:
    pipe.unet = UNetDataParallel(pipe.unet)
    pipe.unet.config, pipe.unet.dtype, pipe.unet.add_embedding = pipe.unet.module.config, pipe.unet.module.dtype, pipe.unet.module.add_embedding
    pipe.to("cuda")
else:
    pipe.to("cuda")

# if using torch < 2.0
# pipe.enable_xformers_memory_efficient_attention()

# pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)


def get_fixed_seed(seed):
    if seed is None or seed == '' or seed <= -1:
        return int(random.randrange(4294967294))
    return int(seed)


def generate_latents(samples, width, height, in_channels, seed_base):
    device = "cuda"
    generator = torch.Generator(device=device)

    latents = None
    seeds = []

    seed_base1 = int(seed_base)
    seed_base1 = get_fixed_seed(seed_base1)

    for i in range(samples):
        # Get a new random seed, store it and use it as the generator state
        seed = seed_base1 + i
        seeds.append(seed)
        generator = generator.manual_seed(seed)
        image_latents = torch.randn(
            (1, in_channels, height // 8, width // 8),
            generator=generator,
            device=device,
            dtype=torch.float16
        )
        latents = image_latents if latents is None else torch.cat(
            (latents, image_latents))
    return latents, seeds


is_gpu_busy = False
def infer(prompt, scale, samples, steps, width, height, seed=-1):
    png_info = PngImagePlugin.PngInfo()
    png_info.add_text(
        "Info",
        f"prompt: {prompt}; CFG: {scale}; Width: {width}; "
        f"Height: {height}; Seed: {seed}; Step: {steps}"
    )

    # prompt, negative = [prompt] * samples, [negative] * samples
    prompt = [prompt] * samples
    g = torch.Generator(device="cuda")
    if seed != -1:
        latents, seeds = generate_latents(
            samples, width, height, pipe.unet.config.in_channels, seed)
    else:
        seed = get_fixed_seed(seed)
        latents, seeds = generate_latents(
            samples, width, height, pipe.unet.config.in_channels, seed)

    
        
    images_b64_list = []

    images = pipe(prompt=prompt, guidance_scale=scale, num_inference_steps=steps, lcm_origin_steps=50, output_type="pil", latents=latents).images
                  
    gc.collect()
    torch.cuda.empty_cache()

    outpath = "output"
    if not os.path.exists(outpath):
        os.makedirs(outpath)
        
    today_date = datetime.today().strftime("%Y-%m-%d")
    new_folder_path = os.path.join(outpath, today_date)
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
    outpath = new_folder_path
    
    for image in images:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        image_b64 = (f"data:image/jpeg;base64,{img_str}")
        images_b64_list.append(image_b64)
    
        grid_count = len(os.listdir(outpath)) - 1
        original_filename = f'grid-{grid_count:04}.png'
        if os.path.exists(os.path.join(outpath, original_filename)):
            timestamp10 = int(time.time())
            original_filename = f'grid-{grid_count:04}-{timestamp10}.png'
        image.save(os.path.join(outpath, original_filename), pnginfo=png_info)
            
    return images_b64_list
    
    
css = """
        .gradio-container {
            font-family: 'IBM Plex Sans', sans-serif;
        }
        .gr-button {
            color: white;
            border-color: black;
            background: black;
        }
        input[type='range'] {
            accent-color: black;
        }
        .dark input[type='range'] {
            accent-color: #dfdfdf;
        }
        .container {
            max-width: 730px;
            margin: auto;
            padding-top: 1.5rem;
        }
        #gallery {
            min-height: 22rem;
            margin-bottom: 15px;
            margin-left: auto;
            margin-right: auto;
            border-bottom-right-radius: .5rem !important;
            border-bottom-left-radius: .5rem !important;
        }
        #gallery>div>.h-full {
            min-height: 20rem;
        }
        .details:hover {
            text-decoration: underline;
        }
        .gr-button {
            white-space: nowrap;
        }
        .gr-button:focus {
            border-color: rgb(147 197 253 / var(--tw-border-opacity));
            outline: none;
            box-shadow: var(--tw-ring-offset-shadow), var(--tw-ring-shadow), var(--tw-shadow, 0 0 #0000);
            --tw-border-opacity: 1;
            --tw-ring-offset-shadow: var(--tw-ring-inset) 0 0 0 var(--tw-ring-offset-width) var(--tw-ring-offset-color);
            --tw-ring-shadow: var(--tw-ring-inset) 0 0 0 calc(3px var(--tw-ring-offset-width)) var(--tw-ring-color);
            --tw-ring-color: rgb(191 219 254 / var(--tw-ring-opacity));
            --tw-ring-opacity: .5;
        }
        #advanced-btn {
            font-size: .7rem !important;
            line-height: 19px;
            margin-top: 12px;
            margin-bottom: 12px;
            padding: 2px 8px;
            border-radius: 14px !important;
        }
        #advanced-options {
            display: none;
            margin-bottom: 20px;
        }
        .footer {
            margin-bottom: 45px;
            margin-top: 35px;
            text-align: center;
            border-bottom: 1px solid #e5e5e5;
        }
        .footer>p {
            font-size: .8rem;
            display: inline-block;
            padding: 0 10px;
            transform: translateY(10px);
            background: white;
        }
        .dark .footer {
            border-color: #303030;
        }
        .dark .footer>p {
            background: #0b0f19;
        }
        .acknowledgments h4{
            margin: 1.25em 0 .25em 0;
            font-weight: bold;
            font-size: 115%;
        }
        .animate-spin {
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            from {
                transform: rotate(0deg);
            }
            to {
                transform: rotate(360deg);
            }
        }
        #share-btn-container {
            display: flex; padding-left: 0.5rem !important; padding-right: 0.5rem !important; background-color: #000000; justify-content: center; align-items: center; border-radius: 9999px !important; width: 13rem;
            margin-top: 10px;
            margin-left: auto;
        }
        #share-btn {
            all: initial; color: #ffffff;font-weight: 600; cursor:pointer; font-family: 'IBM Plex Sans', sans-serif; margin-left: 0.5rem !important; padding-top: 0.25rem !important; padding-bottom: 0.25rem !important;right:0;
        }
        #share-btn * {
            all: unset;
        }
        #share-btn-container div:nth-child(-n+2){
            width: auto !important;
            min-height: 0px !important;
        }
        #share-btn-container .wrap {
            display: none !important;
        }
        
        .gr-form{
            flex: 1 1 50%; border-top-right-radius: 0; border-bottom-right-radius: 0;
        }
        #prompt-container{
            gap: 0;
        }
        #prompt-text-input, #negative-prompt-text-input{padding: .45rem 0.625rem}
        #component-16{border-top-width: 1px!important;margin-top: 1em}
        .image_duplication{position: absolute; width: 100px; left: 50px}
"""

block = gr.Blocks(css=css)

with block:
    gr.HTML(
        """
            <div style="text-align: center; margin: 0 auto;">
              <div
                style="
                  display: inline-flex;
                  align-items: center;
                  gap: 0.8rem;
                  font-size: 1.75rem;
                "
              >
                <svg
                  width="0.65em"
                  height="0.65em"
                  viewBox="0 0 115 115"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <rect width="23" height="23" fill="white"></rect>
                  <rect y="69" width="23" height="23" fill="white"></rect>
                  <rect x="23" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="23" y="69" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="46" width="23" height="23" fill="white"></rect>
                  <rect x="46" y="69" width="23" height="23" fill="white"></rect>
                  <rect x="69" width="23" height="23" fill="black"></rect>
                  <rect x="69" y="69" width="23" height="23" fill="black"></rect>
                  <rect x="92" width="23" height="23" fill="#D9D9D9"></rect>
                  <rect x="92" y="69" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="115" y="46" width="23" height="23" fill="white"></rect>
                  <rect x="115" y="115" width="23" height="23" fill="white"></rect>
                  <rect x="115" y="69" width="23" height="23" fill="#D9D9D9"></rect>
                  <rect x="92" y="46" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="92" y="115" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="92" y="69" width="23" height="23" fill="white"></rect>
                  <rect x="69" y="46" width="23" height="23" fill="white"></rect>
                  <rect x="69" y="115" width="23" height="23" fill="white"></rect>
                  <rect x="69" y="69" width="23" height="23" fill="#D9D9D9"></rect>
                  <rect x="46" y="46" width="23" height="23" fill="black"></rect>
                  <rect x="46" y="115" width="23" height="23" fill="black"></rect>
                  <rect x="46" y="69" width="23" height="23" fill="black"></rect>
                  <rect x="23" y="46" width="23" height="23" fill="#D9D9D9"></rect>
                  <rect x="23" y="115" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="23" y="69" width="23" height="23" fill="black"></rect>
                </svg>
                <h1 style="font-weight: 900; margin-bottom: 7px;margin-top:5px">
                  LCM模型WEBUI - LCM_Dreamshaper_v7
                </h1>
              </div>
              <p style="margin-bottom: 10px; font-size: 94%; line-height: 23px;">
                相关链接：https://github.com/0xbitches/sd-webui-lcm
              </p>
            </div>
        """
    )
    with gr.Group():
        with gr.Box():
            with gr.Row(elem_id="prompt-container").style(mobile_collapse=False, equal_height=True):
                with gr.Column():
                    text = gr.Textbox(
                        label="输入你的词汇",
                        show_label=False,
                        max_lines=1,
                        placeholder="输入你的词汇",
                        elem_id="prompt-text-input",
                    ).style(
                        border=(True, False, True, True),
                        rounded=(True, False, False, True),
                        container=False,
                    )
                btn = gr.Button("生成图片").style(
                    margin=False,
                    rounded=(False, True, True, False),
                    full_width=False,
                )

        gallery = gr.Gallery(
            label="生成图片", show_label=False, elem_id="gallery"
        ).style(grid=[2], height="auto")

        with gr.Group(elem_id="container-advanced-btns"):
            #advanced_button = gr.Button("Advanced options", elem_id="advanced-btn")
            with gr.Group(elem_id="share-btn-container"):
                community_icon = gr.HTML(community_icon_html)
                loading_icon = gr.HTML(loading_icon_html)
                share_button = gr.Button("分享到社区", elem_id="share-btn")

        with gr.Accordion("高级设置", open=False):
        #    gr.Markdown("Advanced settings are temporarily unavailable")
            samples = gr.Slider(label="图片", minimum=1, maximum=max(4, default_num_images), value=default_num_images, step=1)
            steps = gr.Slider(label="步数", minimum=1, maximum=50, value=25, step=1)
            width = gr.Slider(label="宽度", minimum=512, maximum=2048, value=768, step=1)
            height = gr.Slider(label="高度", minimum=512, maximum=2048, value=768, step=1)
            
            guidance_scale = gr.Slider(
                label="CFG", minimum=0, maximum=50, value=9, step=0.1
            )

            seed = gr.Slider(
                label="种子",
                minimum=-1,
                maximum=2147483647,
                step=1,
                randomize=True,
            )


        # negative.submit(infer, inputs=[text, guidance_scale, samples, steps, width, height, seed], outputs=[gallery], postprocess=False)
        text.submit(infer, inputs=[text, guidance_scale, samples, steps, width, height, seed], outputs=[gallery], postprocess=False)
        btn.click(infer, inputs=[text, guidance_scale, samples, steps, width, height, seed], outputs=[gallery], postprocess=False)
        
        #advanced_button.click(
        #    None,
        #    [],
        #    text,
        #    _js="""
        #    () => {
        #        const options = document.querySelector("body > gradio-app").querySelector("#advanced-options");
        #        options.style.display = ["none", ""].includes(options.style.display) ? "flex" : "none";
        #    }""",
        #)
        share_button.click(
            None,
            [],
            [],
            _js=share_js,
        )
        gr.HTML(
            """
                <div class="footer">
                    <p>什么是LCM：潜在一致性模型（Latent Consistency Models，LCMs）潜在一致性模型（LCMs）将潜在空间中的引导反扩散过程视为解决一种增强概率流ODE（PF-ODE）的过程。在这个上下文中，ODE代表"Ordinary Differential Equation"，是一个数学方程，用于描述随时间变化的连续值的演化。PF-ODE则是一种特定类型的ODE，被用来描述信息如何在潜在空间中流动和变化。

LCMs的关键创新在于，它们能够直接预测这种PF-ODE的解，而无需进行多次迭代来逼近解。这意味着LCMs能够更高效地生成图像，因为它们通过直接计算潜在空间中的信息流动，避免了昂贵的迭代过程，从而实现了更快速、高保真度的图像生成。</a>
                    </p>
                </div>
           """
        )
        with gr.Accordion(label="License", open=False):
            gr.HTML(
                """<div class="acknowledgments">
                    <p>仅用于免费应用。by helloworld
                    </p>
               </div>
                """
            )

block.queue().launch(share=share, inbrowser=True)
