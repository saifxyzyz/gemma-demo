# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m")
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-270m")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Your prompt
prompt = "Write a poem on rain"

# Tokenize the input and move it to the device
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Generate new tokens based on the input
outputs = model.generate(**inputs, max_new_tokens=100)

# Decode the generated tokens
response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Print the final response
print(response_text)
