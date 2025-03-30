from tokenizers import ByteLevelBPETokenizer

# 🔹 Eigener Tokenizer mit Byte-Pair-Encoding (BPE)
tokenizer = ByteLevelBPETokenizer()

# 🔹 Beispieltexte zum Training des Tokenizers
texts = [
    "Hallo, wie geht es dir?",
    "Künstliche Intelligenz verändert die Welt.",
    "Neuronale Netzwerke lernen aus Daten.",
]

# 🔹 Tokenizer trainieren
tokenizer.train_from_iterator(texts, vocab_size=5000, min_frequency=2)

# 🔹 Tokenizer speichern
tokenizer.save("models/custom_llm/llm/tokenizer.json")

print("Tokenizer erfolgreich erstellt und gespeichert!")
