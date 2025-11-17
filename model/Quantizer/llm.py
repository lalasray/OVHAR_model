import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load Model & Tokenizer
checkpoint = "HuggingFaceTB/SmolLM-135M-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

# Add custom IMU tokens to the tokenizer (e.g., <imu1>, <imu2>, etc.)
imu_tokens = ["<imu1>", "<imu2>", "<imu3>"]  # Example tokens for IMU data
special_tokens = ["<start_imu>", "<end_imu>"]#, "<activity>"]  # Special tokens for boundaries & activity

# Add special tokens and IMU tokens to tokenizer vocabulary
tokenizer.add_tokens(imu_tokens + special_tokens)  # Adds them to tokenizer vocabulary

# Resize model embeddings to accommodate new tokens
model.resize_token_embeddings(len(tokenizer))

# Resize only the model's input embeddings to 576 dimensions
# This doesn't change the architecture of the transformer layers, just the input embeddings
model.get_input_embeddings().weight.data = torch.nn.Parameter(torch.randn(len(tokenizer), 576))

# Set device for model
model.to(device)

# Your IMU sequence (e.g., imu2, imu1, imu3) + special tokens
imu_sequence = ["<start_imu>", "<imu2>", "<imu1>", "<imu3>", "<end_imu>"]

# Activity context is not needed, we're just asking about the activity directly
input_question = "What activity is being performed with the following IMU data?"

# Combine the question with the IMU sequence
input_sequence = input_question + " " + " ".join(imu_sequence)
print(input_sequence)
# Tokenize the input sequence
input_ids = tokenizer.encode(input_sequence, return_tensors="pt").to(device)

# Print the sequence length
print("Input Sequence Length:", input_ids.shape[1])

# Ensure the input sequence length matches the model's position embedding size
seq_len = input_ids.shape[1]
print("Combined Embedding Sequence Length:", seq_len)

# Create an attention mask to ignore padding tokens (assume no padding, all tokens are real)
attention_mask = torch.ones(input_ids.shape, device=device)

# Get token embeddings for the combined sequence (IMU tokens + question)
with torch.no_grad():
    input_embeddings = model.get_input_embeddings()(input_ids)  # Shape: (1, seq_len, hidden_dim)

# Simulate IMU embeddings (576-dim) corresponding to <imu1>, <imu2>, <imu3>
imu_embedding_data = torch.randn(1, len(imu_tokens), 576).to(device)  # Shape: (1, 3, 576)

# Extract the embeddings for the input question portion
#question_length = len(input_question.split())
#question_embeddings = input_embeddings[:, :question_length]

# Concatenate the IMU embeddings with the question embeddings
combined_embeddings = torch.cat([input_embeddings, imu_embedding_data], dim=1)

# Print the combined embedding sequence length
print("Combined Embedding Sequence Length after concatenation:", combined_embeddings.shape[1])

# Generate Output using the Combined Embeddings
print(attention_mask.shape)
with torch.no_grad():
    outputs = model.generate(inputs_embeds=combined_embeddings, attention_mask=attention_mask, max_new_tokens=50)

# Decode and Print Response
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Model Output:", response)
