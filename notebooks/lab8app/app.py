# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn.functional as F
from model import TransformerClassifier  # we will define this
import pickle
from fastapi import HTTPException

# --- Load vocab ---
with open('imdb_vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

# --- Define model and load checkpoint ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = TransformerClassifier(
    vocab_size=len(vocab),
    embed_dim=32,
    num_heads=4,
    num_encoder_layers=1,
    dim_feedforward=64,
    num_classes=2,
    max_seq_len=512
)
model.load_state_dict(torch.load('transformer_imdb.pth', map_location=device))
model.to(device)
model.eval()

# --- FastAPI setup ---
app = FastAPI()

# --- Define input format ---


class InputText(BaseModel):
    text: str

# --- Helper to tokenize and pad ---


def preprocess(text, pad_idx=vocab['<pad>'], max_seq_len=512):
    tokens = text.split()  # assumes simple space-based tokenization
    token_ids = torch.tensor([vocab.get_stoi().get(token, vocab.get_stoi()[
                             '<unk>']) for token in tokens], dtype=torch.long)

    token_ids = token_ids[:max_seq_len]
    padded = torch.full((max_seq_len,), pad_idx, dtype=torch.long)
    padded[:len(token_ids)] = token_ids

    attention_mask = (padded != pad_idx)

    return padded.unsqueeze(0).to(device), attention_mask.unsqueeze(0).to(device)


# --- Inference Route ---


@app.post("/predict")
def predict(input: InputText):
    try:
        inputs, attention_mask = preprocess(input.text)

        with torch.no_grad():
            outputs = model(inputs, src_key_padding_mask=~attention_mask)
            probs = F.softmax(outputs, dim=1)
            pred = torch.argmax(probs, dim=1).item()

        return {"prediction": pred, "probabilities": probs.squeeze().tolist()}

    except Exception as e:
        print(f"Error during prediction: {e}")  # Log the error in server logs
        raise HTTPException(status_code=500, detail=str(e))
