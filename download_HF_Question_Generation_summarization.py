
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


tokenizer = AutoTokenizer.from_pretrained("Sardar/sql-model-101", use_auth_token=True)
model = AutoModelForSeq2SeqLM.from_pretrained("Sardar/sql-model-101",return_dict=True,torch_dtype=torch.float16, device_map='cpu', use_auth_token=True)


model.save_pretrained('./sql')
tokenizer.save_pretrained('./sql')
