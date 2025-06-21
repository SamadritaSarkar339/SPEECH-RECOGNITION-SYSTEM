# Step 1: Install necessary libraries
!pip install -q transformers librosa soundfile

# Step 2: Import libraries
import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
from IPython.display import Audio

# Step 3: Load pretrained model and tokenizer from Hugging Face
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# Step 4: Load the .wav file and resample to 16000 Hz
speech, rate = librosa.load("/content/drive/MyDrive/hi", sr=16000)

# Optional: Listen to audio
Audio("/content/drive/MyDrive/hi", autoplay=True)

# Step 5: Tokenize the speech input
input_values = tokenizer(speech, return_tensors="pt").input_values

# Step 6: Predict logits (raw scores for characters)
logits = model(input_values).logits

# Step 7: Get predicted token ids
predicted_ids = torch.argmax(logits, dim=-1)

# Step 8: Decode token ids to human-readable text
transcription = tokenizer.decode(predicted_ids[0])
print(transcription)
