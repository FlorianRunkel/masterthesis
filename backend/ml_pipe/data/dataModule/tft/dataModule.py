import pandas as pd
import numpy as np
import pytorch_lightning as pl
from pytorch_forecasting import TimeSeriesDataSet, GroupNormalizer
import json
from rapidfuzz import process, fuzz

from pytorch_forecasting.data.encoders import NaNLabelEncoder

class CareerDataModule(pl.LightningDataModule):
    def __init__(self, dataframe, batch_size=64, max_encoder_length=4, max_prediction_length=2, min_encoder_length=2, min_prediction_length=1):
        super().__init__()
        self.dataframe = dataframe
        self.batch_size = batch_size
        self.max_encoder_length = max_encoder_length
        self.max_prediction_length = max_prediction_length
        self.min_encoder_length = min_encoder_length
        self.min_prediction_length = min_prediction_length
        # Mapping vorbereiten
        with open("/Users/florianrunkel/Documents/02_Uni/04_Masterarbeit/masterthesis/backend/ml_pipe/data/featureEngineering/position_level.json", "r") as f:
            position_entries = json.load(f)
        self.position_map = {
            entry["position"].lower(): (
                entry["level"], 
                entry["branche"],
                entry.get("durchschnittszeit_tage", 365)
            ) for entry in position_entries
        }
        self.all_positions = list(self.position_map.keys())

    def map_position_fuzzy(self, pos, threshold=30):
        pos_clean = pos.lower().strip()
        if pos_clean in self.position_map:
            level, branche, durchschnittszeit = self.position_map[pos_clean]
            match = pos_clean
            score = 100  # Maximale Ähnlichkeit, da exakter Treffer
        else:
            match, score, _ = process.extractOne(pos_clean, self.all_positions, scorer=fuzz.ratio)
            if score >= threshold:
                level, branche, durchschnittszeit = self.position_map[match]
            else:
                return (None, None, None, None)
        return (match, float(level), float(self.get_branche_num(branche)), float(durchschnittszeit))

    def setup(self, stage=None):
        df = self.dataframe.copy()
        df["zeitpunkt"] = pd.to_datetime(df["zeitpunkt"], unit="s")
        df = df.sort_values(["profile_id", "zeitpunkt"])
        df["time_idx"] = df.groupby("profile_id").cumcount()

        # Mapping anwenden
        mapped = df["aktuelle_position"].map(self.map_position_fuzzy)
        df[["mapped_position", "level", "branche", "durchschnittszeit"]] = pd.DataFrame(mapped.tolist(), index=df.index)

        # Numerische Spalten als float casten
        float_cols = [
            "label",
            "berufserfahrung_bis_zeitpunkt",
            "anzahl_wechsel_bisher",
            "anzahl_jobs_bisher",
            "durchschnittsdauer_bisheriger_jobs",
        ]

        for col in float_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

        # Fehlende Werte entfernen
        df = df.dropna(subset=float_cols + ["profile_id", "aktuelle_position"])
        df = df.dropna(subset=["mapped_position", "level", "branche", "durchschnittszeit"])

        # 70/30 Split pro Profil
        def split_profile(group):
            n = len(group)
            split_idx = int(n * 0.7)
            group = group.sort_values("time_idx")
            group["is_train"] = [True]*split_idx + [False]*(n-split_idx)
            return group

        df = df.groupby("profile_id", group_keys=False).apply(split_profile)
        #print(df.groupby("profile_id").size().describe())

        training = df[df["is_train"]].dropna()
        validation = df[~df["is_train"]].dropna()

        # Optional: Profile ohne Trainings- oder Val-Daten entfernen
        valid_profiles = set(training["profile_id"]).intersection(validation["profile_id"])
        self.training = training[training["profile_id"].isin(valid_profiles)]
        self.validation = validation[validation["profile_id"].isin(valid_profiles)]

        print("Train:", len(self.training), "Val:", len(self.validation))
        print(self.validation.head(20))

        # TimeSeriesDataSet für Training
        self.training_dataset = TimeSeriesDataSet(
            self.training,
            time_idx="time_idx",
            target="label",
            group_ids=["profile_id"],
            max_encoder_length=self.max_encoder_length,
            max_prediction_length=self.max_prediction_length,
            min_encoder_length=self.min_encoder_length,
            min_prediction_length=self.min_prediction_length,
            time_varying_unknown_reals=[
                "berufserfahrung_bis_zeitpunkt",
                "anzahl_wechsel_bisher",
                "anzahl_jobs_bisher",
                "durchschnittsdauer_bisheriger_jobs",
                "level",
                "branche",
                "durchschnittszeit"
            ],
            time_varying_known_reals=[
                "time_idx"
            ],
            static_categoricals=["mapped_position"],
            categorical_encoders={
                "profile_id": NaNLabelEncoder(add_nan=True),
                "mapped_position": NaNLabelEncoder(add_nan=True),
            }, 
            target_normalizer=GroupNormalizer(groups=["profile_id"], transformation="softplus"),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True,
        )
        import torch
        torch.save(self.training_dataset, "training_dataset.pt")

        # TimeSeriesDataSet für Validation
        self.validation_dataset = TimeSeriesDataSet.from_dataset(
            self.training_dataset,
            self.validation,
            predict=True,
            stop_randomization=True,
            allow_missing_timesteps=True,    
        )

    def get_branche_num(self, branche):
        """Konvertiert Branchennamen in numerische Werte."""
        branche_map = {"sales": 1, "engineering": 2, "consulting": 3}
        return branche_map.get(branche, 0)

    def train_dataloader(self):
        return self.training_dataset.to_dataloader(train=True, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return self.validation_dataset.to_dataloader(train=False, batch_size=self.batch_size * 2, num_workers=4)