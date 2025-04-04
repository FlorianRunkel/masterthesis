from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

def get_callbacks(config):
    """
    Erstellt die Callbacks für das Training basierend auf der Konfiguration
    
    Args:
        config (dict): Konfigurationsdictionary
        
    Returns:
        list: Liste der Callbacks
    """
    callbacks = []
    
    # Early Stopping
    early_stopping = EarlyStopping(
        monitor=config['callbacks']['early_stopping']['monitor'],
        patience=config['callbacks']['early_stopping']['patience'],
        mode=config['callbacks']['early_stopping']['mode']
    )
    callbacks.append(early_stopping)
    
    # Model Checkpoint
    model_checkpoint = ModelCheckpoint(
        monitor=config['callbacks']['model_checkpoint']['monitor'],
        mode=config['callbacks']['model_checkpoint']['mode'],
        save_top_k=config['callbacks']['model_checkpoint']['save_top_k'],
        filename=config['callbacks']['model_checkpoint']['filename']
    )
    callbacks.append(model_checkpoint)
    
    return callbacks 