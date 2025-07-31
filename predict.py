import os
from typing import List
from PIL import Image
import torch
# FIX: Import Path and Input from 'cog'
from cog import BasePredictor, Input, Path

from diffusers import StableDiffusionXLInstantIDPipeline, ControlNetModel
from insightface.app import FaceAnalysis

class Predictor(BasePredictor):
    def setup(self):
        print("Loading model...")
        self.face_app = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))
        controlnet = ControlNetModel.from_pretrained("InstantX/InstantID", subfolder="ControlNetModel", torch_dtype=torch.float16)
        base_model = 'wangqixun/YamerMIX_v8'
        self.pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(base_model, controlnet=controlnet, torch_dtype=torch.float16).to("cuda")
        self.pipe.load_ip_adapter_instantid("InstantX/InstantID", subfolder="ip-adapter")
        print("Model loaded successfully!")

    def predict(
        self,
        image: Path = Input(description="Input image of a face"),
        prompt: str = Input(description="The prompt for the storybook style"),
        negative_prompt: str = Input(default="blurry, low quality, nsfw, text, words, letters", description="Items to exclude from the image"),
        num_inference_steps: int = Input(default=30, description="Number of inference steps"),
        guidance_scale: float = Input(default=5.0, description="Guidance scale"),
        ip_adapter_scale: float = Input(default=0.8, description="Strength of the face likeness"),
    ) -> Path:
        print("Starting prediction...")
        face_image = Image.open(str(image)).convert("RGB")
        face_info = self.face_app.get(face_image)
        if len(face_info) == 0:
            raise ValueError("No face detected in the input image.")
        face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]
        face_embedding = face_info.normed_embedding
        self.pipe.set_ip_adapter_scale(ip_adapter_scale)
        result_image = self.pipe(
            prompt,
            negative_prompt=negative_prompt,
            ip_adapter_image_embeds=[face_embedding],
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]
        output_path = "/tmp/output.png"
        result_image.save(output_path)
        print("Prediction finished!")
        return Path(output_path)
