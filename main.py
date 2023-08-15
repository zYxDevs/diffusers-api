import uvicorn
import torch
import base64

from torch import autocast
from PIL import Image
from fastapi import FastAPI, HTTPException
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from Schemas import Txt2ImgSchemas

pipe = StableDiffusionPipeline.from_pretrained(
  "hakurei/waifu-diffusion",
  use_safetensors=True,
  torch_dtype=torch.float16
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
  pipe.scheduler.config,
  use_karras_sigmas=True
)
pipe.to("cuda")
#pipe.load_lora_weights("LoRa", weight_name="light_and_shadow.safetensors")


def createimage(prompt, negative_prompt):
  image = pipe(
    prompt,
    negative_prompt=negative_prompt,
    width=512,
    height=512,
    guidance_scale=12,
    target_size=(1024,1024),
    original_size=(4096,4096),
    num_inference_steps=50
  ).images[0]
  image.save(".config/image.png")
  with open(".config/image.png", "r") as file:
    base_64 = base64.b64encode(file).decode("utf-8")
  return base_64
  

app = FastAPI()

@app.post("/api/txt2img/")
def txt2img(data: Txt2ImgSchemas):
  try:
    prompt: data.prompt
    negative_prompt: data.negative_prompt
    resolve = createimage(prompt, negative_prompt)
    
    return {
      "images": [resolve]
    }
  except Exception as error:
    raise HTTPException(status_code=404, detail=str(error))



if __name__ == "__main__":
  uvicorn.run("main:app", host="0.0.0.0", port=3000)