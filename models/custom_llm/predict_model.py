import torch
from models.custom_llm import SimpleGPT
from transformers import PreTrainedTokenizerFast

# Funktionen zum Laden des Tokenizers und des Modells
def load_tokenizer(tokenizer_path):
    return PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

def load_model_and_tokenizer(model_class, vocab_size, model_path, tokenizer_path):
    model = model_class(vocab_size)
    model.load_state_dict(torch.load(model_path))
    tokenizer = load_tokenizer(tokenizer_path)
    model.eval()
    return model, tokenizer

def generate_custom_llm(first_name, last_name, location, experiences, max_length=128):

    # prompt
    prompt= getPrompt(first_name, last_name, location, experiences)

    # Modell- und Tokenizer-Pfade
    model_path = "models/custom_llm/llm/custom_llm.pth"
    tokenizer_path = "models/custom_llm/llm"

    VOCAB_SIZE = 5000
    model, tokenizer = load_model_and_tokenizer(SimpleGPT, VOCAB_SIZE, model_path, tokenizer_path)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length, padding="max_length").to("cuda" if torch.cuda.is_available() else "cpu")
    
    with torch.no_grad():
        output = model(inputs["input_ids"])
    
    predicted_ids = torch.argmax(output, dim=-1)
    predicted_text = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
    predicted_text = predicted_text.replace(tokenizer.pad_token, "").strip()
    
    return predicted_text


def getPrompt(first_name, last_name, location, experiences): 
    prompt = f"""
    Du bist eine KI, die Karriereempfehlungen basierend auf bisherigen Erfahrungen gibt.

    **Name:** {first_name} {last_name}  
    **Wohnort:** {location}  
    **Berufserfahrung:**  
    {experiences}

    **Frage:** Wann wäre der optimale Zeitpunkt für den nächsten Karriereschritt und welche Position wäre empfehlenswert?

    **Antwort:**  
    Gebe ausschließlich das Datum (Monat/Jahr) und die empfohlene Position aus.  
    Falls keine Vorhersage möglich ist, antworte mit: "Aktuell keine Vorhersage möglich."
    """
    return prompt