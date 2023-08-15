from pydantic import BaseModel

class Txt2ImgSchemas(BaseModel):
  prompt: str,
  negative_prompt: str = None