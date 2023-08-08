from fastapi import Request,FastAPI
from pydantic import BaseModel
import uvicorn

import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


app = FastAPI()

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'


model_id="./sql"
tokenizer = AutoTokenizer.from_pretrained(model_id)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
model = AutoModelForCausalLM.from_pretrained(model_id, return_dict=True,quantization_config=bnb_config, device_map={"":0})
generation_config=model.generation_config
generation_config.max_new_tokens=256
generation_config.temperature=0.2
generation_config.do_sample=True
generation_config.top_p=0.95
generation_config.num_return_sequences=1
generation_config.eos_token_id=49155
generation_config.pad_token_id=49155

class SummaryRequest(BaseModel):
    text: str
  

def get_sql(t,tokenizer,model):
  txt = t['text']
 
  prompt=f"""<|system|>\n'Write a SQL query for the given Question. Schema contains information about the database including table names delimited by "|" and columns of a table delimited by ",". Schema also contains info about primary and foreign keys.'<|end|>\n
  <|user|>\n{txt}<|end|>\n
  <|assisstant|>"""
  encoding=tokenizer(prompt,return_tensors='pt').to('cuda')
  with torch.inference_mode():
      outputs=model.generate(input_ids=encoding.input_ids,
                        attention_mask=encoding.attention_mask,
                        generation_config=generation_config)
  output = tokenizer.decode(outputs[0],skip_special_tokens=True).split("\n")[-1]
  return {'sql':output}


@app.get('/')
async def home():
    return {"message": "Hello World"}

@app.post("/summary")
async def getsummary(user_request_in: SummaryRequest):
    payload = {"text":user_request_in.text}
    summ = get_sql(payload,tokenizer,model)
    summ["Device"]= torch_device
    return summ


