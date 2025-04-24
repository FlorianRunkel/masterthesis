import torch
from models.tft.model import TFTModel

def predict_career(sequence_features, global_features, model_path="saved_models/career_lstm_20250424_140701.pt"):
    """
    Macht eine Vorhersage mit dem TFTModel
    
    Args:
        sequence_features (list): Liste von Listen mit Sequenz-Features [seq_len, 13]
            - Position Features (5): [level, branche, duration_months, time_since_start, time_until_end]
            - Transition Features (8): [gap_months, level_change, internal_move, location_change, 
                                      branche_change, previous_level, previous_branche, previous_duration]
        global_features (list): Liste mit globalen Features [9]
            - [highest_degree, age_category, total_experience_years, avg_position_gap,
               internal_moves_ratio, location_changes_ratio, branche_changes_ratio,
               avg_level_change, positive_moves_ratio]
    """
    # Lade den Checkpoint zuerst
    checkpoint = torch.load(model_path)
    
    # Extrahiere die Hyperparameter aus dem Checkpoint
    hyperparameters = checkpoint.get('hyperparameters', {})
    
    # Modell mit den gespeicherten Hyperparametern initialisieren
    model = TFTModel(
        sequence_features=hyperparameters.get('sequence_dim', 13),
        global_features=hyperparameters.get('global_dim', 9),
        hidden_size=hyperparameters.get('hidden_size', 128),
        num_layers=hyperparameters.get('num_layers', 2),
        dropout=hyperparameters.get('dropout', 0.2),
        bidirectional=hyperparameters.get('bidirectional', True)
    )
    
    # Lade die Modellgewichte
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()

    # Input vorbereiten
    seq_tensor = torch.tensor(sequence_features, dtype=torch.float32).unsqueeze(0)  # [1, seq_len, 13]
    global_tensor = torch.tensor(global_features, dtype=torch.float32).unsqueeze(0)  # [1, 9]
    lengths = torch.tensor([len(sequence_features)]).long()  # [1]
    
    # Vorhersage machen
    with torch.no_grad():
        pred = model((seq_tensor, global_tensor, lengths))
    
    # Vorhersage interpretieren
    pred_value = float(pred.item())
    
    # Interpretation der Vorhersage
    if pred_value > 0.7:
        status = "sehr wahrscheinlich wechselbereit"
    elif pred_value > 0.5:
        status = "wahrscheinlich wechselbereit"
    elif pred_value > 0.3:
        status = "möglicherweise wechselbereit"
    else:
        status = "bleibt wahrscheinlich"
    
    return pred_value, status

if __name__ == "__main__":
    # Beispiel für die Verwendung
    sequence_features = [
        # Position 1: [level, branche, duration_months, time_since_start, time_until_end,
        #              gap_months, level_change, internal_move, location_change, 
        #              branche_change, previous_level, previous_branche, previous_duration]
        [2, 1, 24, 0, 24,  # Position Features
         0, 0, 0, 0, 0, 2, 1, 24]  # Transition Features
    ]

    # Globale Features: [highest_degree, age_category, total_experience_years, avg_position_gap,
    #                    internal_moves_ratio, location_changes_ratio, branche_changes_ratio,
    #                    avg_level_change, positive_moves_ratio]
    global_features = [2, 2, 5, 0, 0, 0, 0, 0, 0.5]

    pred_value, status = predict_career(sequence_features, global_features)
    print(f"Vorhersage: {pred_value:.2f} - {status}") 