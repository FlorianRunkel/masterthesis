import pytorch_lightning as pl
from pytorch_forecasting import TemporalFusionTransformer, QuantileLoss

class TFTModel(pl.LightningModule):
    def __init__(self, training_dataset, learning_rate=0.03, hidden_size=32, attention_head_size=2,
                 dropout=0.1, hidden_continuous_size=16, output_size=7, **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=["training_dataset"])
        self.tft = TemporalFusionTransformer.from_dataset(
            training_dataset,
            learning_rate=learning_rate,
            hidden_size=hidden_size,
            attention_head_size=attention_head_size,
            dropout=dropout,
            hidden_continuous_size=hidden_continuous_size,
            output_size=output_size,
            loss=QuantileLoss(),
            log_interval=10,
            reduce_on_plateau_patience=4,
            **kwargs
        )
        print(f"Number of parameters in network: {self.tft.size()/1e3:.1f}k")

    def forward(self, *args, **kwargs):
        return self.tft(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        return self.tft.training_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.tft.validation_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.tft.test_step(batch, batch_idx)

    def configure_optimizers(self):
        return self.tft.configure_optimizers()

    def predict(self, *args, **kwargs):
        return self.tft.predict(*args, **kwargs)

    def interpret_output(self, *args, **kwargs):
        return self.tft.interpret_output(*args, **kwargs)

    def plot_prediction(self, *args, **kwargs):
        return self.tft.plot_prediction(*args, **kwargs)

    def plot_interpretation(self, *args, **kwargs):
        return self.tft.plot_interpretation(*args, **kwargs)