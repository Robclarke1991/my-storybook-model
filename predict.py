import os
from typing import List
from PIL import Image
import torch
# FIX 1: Import the special 'Path' object directly from 'cog'
from cog import Path
# FIX 2: Import BasePredictor from 'cog.predictor'
from cog.predictor import BasePredictor

# Custom pipeline script
from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline

# Import the ControlNetModel
from diffusers import ControlNetModel
from insightface.app import FaceAnalysis


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading model...")

        # Initialize the face analysis model
        self.face_app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))

        # Load the ControlNet model from the Hugging Face Hub
        controlnet = ControlNetModel.from_pretrained(
            "InstantX/InstantID",
            subfolder="ControlNetModel",
            torch_dtype=torch.float16
        )

        # Load the base model and create the pipeline
        base_model = 'wangqixun/YamerMIX_v8'
        self.pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
            base_model,
            controlnet=controlnet,
            torch_dtype=torch.float16
        ).to("cuda")

        # Load the IP Adapter model from the Hugging Face Hub
        self.pipe.load_ip_adapter("InstantX/InstantID", subfolder="ip-adapter", weight_name="ip-adapter.bin")
        print("Model loaded successfully!")

    def predict(
        self,
        image: Path,
        prompt: str,
        negative_prompt: str = "blurry, low quality, nsfw, text, words, letters",
        num_inference_steps: int = 30,
        guidance_scale: float = 5.0,
        ip_adapter_scale: float = 0.8,

    ) -> Path:
        """Run a single prediction on the model"""
        print("Starting prediction...")
        
        # Process the input image
        face_image = Image.open(str(image)).convert("RGB")
        face_info = self.face_app.get(face_image)
        if len(face_info) == 0:
            raise ValueError("No face detected in the input image.")
        
        # Use the first detected face
        face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]
        face_embedding = face_info.normed_embedding

        self.pipe.set_ip_adapter_scale(ip_adapter_scale)

        # Generate the image
        result_image = self.pipe(
            prompt,
            negative_prompt=negative_prompt,
            ip_adapter_image_embeds=[face_embedding],
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]

        # Save the output image to a temporary file
        output_path = Path("/tmp/output.png")
        result_image.save(output_path)
        print("Prediction finished!")

        return output_path
