# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from PIL import Image

import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin, IPAdapterMixin, LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, ControlNetModel, UNet2DConditionModel
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    deprecate,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import is_compiled_module, is_torch_version

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import StableDiffusionXLInstantIDPipeline, T2IAdapter, UNet2DConditionModel, EulerAncestralDiscreteScheduler
        >>> from diffusers.utils import load_image
        >>> from controlnet_aux importDWposeDetector
        >>> from PIL import Image

        >>> # DWpose detector
        >>> detector = DWposeDetector.from_pretrained("yzd-v/DWPose")

        >>> # Load InstantID pipeline
        >>> controlnet = T2IAdapter.from_pretrained(
        ...     "InstantX/InstantID", subfolder="ControlNet", torch_dtype=torch.float16
        ... )
        >>> unet = UNet2DConditionModel.from_pretrained(
        ...     "InstantX/sdxl-unet-base", torch_dtype=torch.float1h
        ... )
        >>> pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
        ...     "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet, unet=unet, torch_dtype=torch.float16
        ... )
        >>> pipe.load_ip_adapter_instantid("InstantX/InstantID", subfolder="ip-adapter")
        >>> pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        >>> pipe.cuda()

        >>> # Prepare face image
        >>> face_image = load_image("[https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_einstein.png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_einstein.png)")
        >>> face_image.resize((224, 224))

        >>> # Prepare pose image
        >>> pose_image = load_image("[https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/person.png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/person.png)")
        >>> pose_image = detector(pose_image)


        >>> prompt = "a person in a purple dress is dancing"
        >>> negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        >>> # Generate image
        >>> image = pipe(
        ...     prompt,
        ...     image=pose_image,
        ...     ip_adapter_image=face_image,
        ...     negative_prompt=negative_prompt,
        ...     num_inference_steps=30,
        ...     guidance_scale=7.5,
        ... ).images[0]
        ```
"""


def draw_kps(image_pil, kps, color_list=[(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255)]):
    stickwidth = 4
    limbSeq = np.array([[0, 2], [1, 8], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9], [9, 10]])
    kps = np.array(kps)

    w, h = image_pil.size
    out_img = np.zeros([h, w, 3])

    for i in range(len(limbSeq)):
        index = limbSeq[i]
        color = color_list[i]

        x = kps[index][:, 0]
        y = kps[index][:, 1]
        
        length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
        polygon = cv2.ellipse2Poly((int(np.mean(x)), int(np.mean(y))), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        out_img = cv2.fillConvexPoly(out_img.copy(), polygon, color)
    out_img = (out_img * 0.6).astype(np.uint8)

    for idx_kp, kp in enumerate(kps):
        color = color_list[idx_kp]
        x, y = kp
        out_img = cv2.circle(out_img.copy(), (int(x), int(y)), stickwidth * 2, color, -1)

    out_img_pil = Image.fromarray(out_img.astype(np.uint8))
    return out_img_pil

class StableDiffusionXLInstantIDPipeline(
    DiffusionPipeline, TextualInversionLoaderMixin, LoraLoaderMixin, FromSingleFileMixin, IPAdapterMixin
):
    model_cpu_offload_seq = "text_encoder->text_encoder_2->unet->vae"
    _optional_components = []
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds", "add_text_embeds", "add_time_ids", "negative_add_time_ids"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        text_encoder_2: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        tokenizer_2: CLIPTokenizer,
        unet: UNet2DConditionModel,
        controlnet: ControlNetModel,
        scheduler: KarrasDiffusionSchedulers,
        image_encoder: Any = None,
        feature_extractor: CLIPImageProcessor = None,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=unet,
            controlnet=controlnet,
            scheduler=scheduler,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.controlnet_conditioning_scale = 1.0

    def set_ip_adapter_scale(self, scale):
        for attn_processor in self.unet.attn_processors.values():
            if isinstance(attn_processor, IPAdapterAttnProcessor):
                attn_processor.scale = scale

    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    ):
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale
            if self.text_encoder is not None:
                adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)
            if self.text_encoder_2 is not None:
                adjust_lora_scale_text_encoder(self.text_encoder_2, lora_scale)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        tokenizers = [self.tokenizer, self.tokenizer_2] if self.tokenizer is not None else [self.tokenizer_2]
        text_encoders = (
            [self.text_encoder, self.text_encoder_2] if self.text_encoder is not None else [self.text_encoder_2]
        )

        if prompt_embeds is None:
            prompt_embeds_list = []
            for tokenizer, text_encoder in zip(tokenizers, text_encoders):
                if isinstance(self, TextualInversionLoaderMixin):
                    prompt = self.maybe_convert_prompt(prompt, tokenizer)

                text_inputs = tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                text_input_ids = text_inputs.input_ids
                untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

                if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                    text_input_ids, untruncated_ids
                ):
                    removed_text = tokenizer.batch_decode(
                        untruncated_ids[:, tokenizer.model_max_length - 1 : -1]
                    )
                    logger.warning(
                        "The following part of your input was truncated because CLIP's maximum input length is "
                        f"{tokenizer.model_max_length} tokens: {removed_text}"
                    )

                prompt_embeds = text_encoder(
                    text_input_ids.to(device),
                    output_hidden_states=True,
                )

                pooled_prompt_embeds = prompt_embeds[0]
                if clip_skip is None:
                    prompt_embeds = prompt_embeds.hidden_states[-2]
                else:
                    prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]

                prompt_embeds_list.append(prompt_embeds)

            prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

        zero_out_negative_prompt = negative_prompt is None and do_classifier_free_guidance

        if do_classifier_free_guidance and negative_prompt_embeds is None and pooled_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            uncond_tokens: List[str]
            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt] * batch_size
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            negative_prompt_embeds_list = []
            for tokenizer, text_encoder in zip(tokenizers, text_encoders):
                if isinstance(self, TextualInversionLoaderMixin):
                    uncond_tokens = self.maybe_convert_prompt(uncond_tokens, tokenizer)

                max_length = prompt_embeds.shape[1]
                uncond_input = tokenizer(
                    uncond_tokens,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                negative_prompt_embeds = text_encoder(
                    uncond_input.input_ids.to(device),
                    output_hidden_states=True,
                )
                negative_pooled_prompt_embeds = negative_prompt_embeds[0]
                negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]

                negative_prompt_embeds_list.append(negative_prompt_embeds)

            negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)

        if do_classifier_free_guidance:
            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

            if self.text_encoder is not None:
                if pooled_prompt_embeds is None:
                    raise ValueError("pooled_prompt_embeds must be specified if do_classifier_free_guidance is True and text_encoder is not None")
                if negative_pooled_prompt_embeds is None:
                    raise ValueError("negative_pooled_prompt_embeds must be specified if do_classifier_free_guidance is True and text_encoder is not None")

            if negative_pooled_prompt_embeds is not None:
                negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
                negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(1, num_images_per_prompt)
                negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.view(batch_size * num_images_per_prompt, -1)
        
        if prompt_embeds is not None:
            prompt_embeds = prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)

        if pooled_prompt_embeds is not None:
            pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)

        return prompt_embeds, pooled_prompt_embeds, negative_prompt_embeds, negative_pooled_prompt_embeds

    def _encode_image(self, image, device, num_images_per_prompt, do_classifier_free_guidance):
        dtype = next(self.image_encoder.parameters()).dtype
        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor(image, return_tensors="pt").pixel_values
        image = image.to(device=device, dtype=dtype)
        image_embeds = self.image_encoder(image, output_hidden_states=True).hidden_states[-2]
        bs_embed, seq_len, _ = image_embeds.shape
        image_embeds = image_embeds.repeat(1, num_images_per_prompt)
        image_embeds = image_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
        if do_classifier_free_guidance:
            negative_image_embeds = torch.zeros_like(image_embeds)
            image_embeds = torch.cat([negative_image_embeds, image_embeds])
        return image_embeds

    def prepare_extra_step_kwargs(self, generator, eta):
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt, image, height, width, callback_steps,
        negative_prompt=None, prompt_embeds=None, negative_prompt_embeds=None,
        ip_adapter_image_embeds=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")
        if (callback_steps is None) or (callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)):
            raise ValueError(f"`callback_steps` has to be a positive integer but is {callback_steps} of type {type(callback_steps)}.")
        if prompt is not None and prompt_embeds is not None:
            raise ValueError("Cannot forward both `prompt` and `prompt_embeds`. Please make sure to only forward one of the two.")
        elif prompt is None and prompt_embeds is None:
            raise ValueError("Provide either `prompt` or `prompt_embeds`.")
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError("Cannot forward both `negative_prompt` and `negative_prompt_embeds`.")
        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError("`prompt_embeds` and `negative_prompt_embeds` must have the same shape.")
        if ip_adapter_image_embeds is not None:
            if isinstance(ip_adapter_image_embeds, list) and any(e.ndim != 3 for e in ip_adapter_image_embeds):
                raise ValueError("Each `ip_adapter_image_embeds` must be a 3D tensor.")
            elif ip_adapter_image_embeds.ndim != 3:
                raise ValueError("`ip_adapter_image_embeds` must be a 3D tensor.")

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(f"You have passed a list of generators of length {len(generator)}, but requested an effective batch size of {batch_size}.")
        if latents is None:
            latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_control_image(
        self, image, width, height, batch_size, num_images_per_prompt,
        device, dtype, do_classifier_free_guidance=False,
    ):
        image = self.image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)
        image_batch_size = image.shape[0]
        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            repeat_by = num_images_per_prompt
        image = image.repeat_interleave(repeat_by, dim=0)
        image = image.to(device=device, dtype=dtype)
        if do_classifier_free_guidance:
            image = torch.cat([image] * 2)
        return image

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        image: Optional[Union[Image.Image, List[Image.Image]]] = None,
        ip_adapter_image: Optional[Union[Image.Image, List[Image.Image]]] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 20,
        guidance_scale: float = 5.0,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_conditioning_scale: float = 1.0,
        ip_adapter_scale: float = 1.0,
        clip_skip: Optional[int] = None,
    ):
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        self.check_inputs(prompt, image, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds, ip_adapter_image_embeds)
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0
        (prompt_embeds, pooled_prompt_embeds, negative_prompt_embeds, negative_pooled_prompt_embeds) = self._encode_prompt(
            prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt,
            prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds, negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=None, clip_skip=clip_skip,
        )
        if ip_adapter_image is not None:
            image_embeds = self._encode_image(ip_adapter_image, device, batch_size*num_images_per_prompt, do_classifier_free_guidance)
        else:
            if ip_adapter_image_embeds is None:
                raise ValueError("`ip_adapter_image` and `ip_adapter_image_embeds` cannot be both None.")
            if do_classifier_free_guidance:
                image_embeds = torch.cat([torch.zeros_like(ip_adapter_image_embeds), ip_adapter_image_embeds], dim=0)
            else:
                image_embeds = ip_adapter_image_embeds
        control_image = self.prepare_control_image(
            image=image, width=width, height=height, batch_size=batch_size * num_images_per_prompt,
            num_images_per_prompt=num_images_per_prompt, device=device, dtype=self.controlnet.dtype,
            do_classifier_free_guidance=do_classifier_free_guidance,
        )
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt, num_channels_latents, height, width,
            prompt_embeds.dtype, device, generator, latents,
        )
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        add_time_ids = self._get_add_time_ids(
            height, width, 0, 0, batch_size * num_images_per_prompt, prompt_embeds.dtype
        )
        if do_classifier_free_guidance:
            add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)
        add_time_ids = add_time_ids.to(device)
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self.set_ip_adapter_scale(ip_adapter_scale)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_.scale_model_input(latent_model_input, t)
                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    latent_model_input, t,
                    encoder_hidden_states=prompt_embeds if not do_classifier_free_guidance else torch.cat([negative_prompt_embeds, prompt_embeds]),
                    controlnet_cond=control_image, conditioning_scale=controlnet_conditioning_scale, return_dict=False,
                )
                if do_classifier_free_guidance:
                    prompt_embeds_input = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                    pooled_prompt_embeds_input = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
                else:
                    prompt_embeds_input = prompt_embeds
                    pooled_prompt_embeds_input = pooled_prompt_embeds
                added_cond_kwargs = {"text_embeds": pooled_prompt_embeds_input, "time_ids": add_time_ids}
                noise_pred = self.unet(
                    latent_model_input, t, encoder_hidden_states=prompt_embeds_input,
                    cross_attention_kwargs=cross_attention_kwargs,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    added_cond_kwargs=added_cond_kwargs,
                    encoder_hidden_states_ip_adapter=image_embeds, return_dict=False,
                )[0]
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // self.scheduler.order
                        callback(step_idx, t, latents)
        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.scaling_factor, return_dict=False)[0]
        else:
            image = latents
        do_convert_rgb = self.vae.config.force_upcast and image.dtype != torch.float32
        if do_convert_rgb:
            image = image.float()
        image = self.image_processor.postprocess(image, output_type=output_type)
        self.maybe_free_model_hooks()
        if not return_dict:
            return (image,)
        return StableDiffusionXLPipelineOutput(images=image)
    
    def _get_add_time_ids(
        self, original_size, crops_coords_top_left, target_size, aesthetic_score, negative_aesthetic_score, dtype
    ):
        if self.text_encoder_2 is None:
            return None
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        if self.text_encoder_2.config.projection_dim is not None:
            add_aesthetic_score = torch.tensor([aesthetic_score], dtype=dtype)
            add_negative_aesthetic_score = torch.tensor([negative_aesthetic_score], dtype=dtype)
            add_time_ids = torch.cat(
                [add_time_ids, add_aesthetic_score, add_negative_aesthetic_score]
            )
        add_time_ids = add_time_ids.unsqueeze(0)
        return add_time_ids
