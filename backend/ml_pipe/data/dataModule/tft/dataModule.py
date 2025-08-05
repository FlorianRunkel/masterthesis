import pandas as pd
import numpy as np
import pytorch_lightning as pl
from pytorch_forecasting import TimeSeriesDataSet, GroupNormalizer
import torch
from backend.ml_pipe.data.featureEngineering.tft.feature_engineering_tft import FeatureEngineering
from pytorch_forecasting.data.encoders import NaNLabelEncoder
import json
from rapidfuzz import process, fuzz
import os
import logging

'''
DataModule for TFT
'''
class DataModule(pl.LightningDataModule):

    '''
    Initialize DataModule
    '''
    def __init__(self, dataframe, batch_size=64, max_encoder_length=4, max_prediction_length=2, min_encoder_length=2, min_prediction_length=1):
        super().__init__()
        self.dataframe = dataframe
        self.batch_size = batch_size
        self.max_encoder_length = max_encoder_length
        self.max_prediction_length = max_prediction_length
        self.min_encoder_length = min_encoder_length
        self.min_prediction_length = min_prediction_length

        self.fe_tft = FeatureEngineering()

        script_dir = os.path.dirname(__file__)
        position_level_path = os.path.join(script_dir, '..', '..', 'featureEngineering', 'position_level.json')
        with open(position_level_path, "r") as f:
            position_entries = json.load(f)
        self.position_map = {
            entry["position"].lower(): (
                entry["level"], 
                entry["branche"],
                entry.get("durchschnittszeit_tage", 365)
            ) for entry in position_entries
        }
        self.all_positions = list(self.position_map.keys())

    '''
    Map position to level, branch and average time
    '''
    def map_position_fuzzy(self, pos, threshold=30):
        pos_clean = pos.lower().strip()
        if pos_clean in self.position_map:
            level, branche, durchschnittszeit = self.position_map[pos_clean]
            match = pos_clean
            score = 100
        else:
            match, score, _ = process.extractOne(pos_clean, self.all_positions, scorer=fuzz.ratio)
            if score >= threshold:
                level, branche, durchschnittszeit = self.position_map[match]
            else:
                return (None, None, None, None)
        return (match, float(level), float(self.get_branche_num(branche)), float(durchschnittszeit))

    '''
    Setup data for training, validation and testing
    '''
    def setup(self, stage=None):
        logging.info("Setting up TFT DataModule with extended features...")

        docs = self.dataframe.to_dict('records')
        logging.info(f"Number of documents: {len(docs)}")

        sequences, labels, positions = self.fe_tft.extract_sequences_by_profile(docs, min_seq_len=2)

        data = []
        debug_count = 0
        filtered_count = 0
        total_count = 0

        for i, (seq, label, pos_seq) in enumerate(zip(sequences, labels, positions)):
            for j, (features, position) in enumerate(zip(seq, pos_seq)):
                total_count += 1

                features_numeric = [float(f.item()) if hasattr(f, 'item') else float(f) for f in features]
                label_numeric = float(label.item()) if hasattr(label, 'item') else float(label)

                feature_sum = sum(features_numeric)
                if feature_sum == 0:
                    filtered_count += 1
                    continue

                if debug_count < 5:
                    print(f"DEBUG - Profile {i}, Time {j}:")
                    print(f"  Features (erste 10): {features_numeric[:10]}")
                    print(f"  Label: {label_numeric}")
                    print(f"  Position: {position}")
                    print(f"  Feature-Summe: {sum(features_numeric)}")
                    debug_count += 1

                data.append({
                    'profile_id': i,
                    'time_idx': j,
                    'target': label_numeric,
                    'position': position,
                    **{f'feature_{k}': v for k, v in enumerate(features_numeric)}
                })

        print(f"DEBUG - Filterung:")
        print(f"  Gesamte Zeilen: {total_count}")
        print(f"  Gefilterte Zeilen (alle Features = 0): {filtered_count}")
        print(f"  Verbleibende Zeilen: {len(data)}")

        df = pd.DataFrame(data)
        print(f"DataFrame Shape: {df.shape}")
        print(f"Anzahl Profile: {df['profile_id'].nunique()}")
        print(f"Feature Spalten: {[col for col in df.columns if col.startswith('feature_')]}")

        feature_names = {
            # Basis-Features
            'feature_0': 'berufserfahrung_tage',
            'feature_1': 'anzahl_wechsel_bisher',
            'feature_2': 'anzahl_jobs_bisher',
            'feature_3': 'durchschnittsdauer_jobs',
            #'feature_4': 'highest_degree',
            'feature_4': 'age_category',

            # Position-Features
            'feature_5': 'position_level',
            'feature_6': 'position_branche',
            'feature_7': 'position_durchschnittszeit',

            # Position-ID
            'feature_8': 'position_id_numeric',

            # Time-Features
            'feature_9': 'weekday',
            'feature_10': 'weekday_sin',
            'feature_11': 'weekday_cos',
            'feature_12': 'month',
            'feature_13': 'month_sin',
            'feature_14': 'month_cos',

            # Career path features - last 2 positions
            'feature_15': 'prev_position_1_level',
            'feature_16': 'prev_position_1_branche',
            'feature_17': 'prev_position_1_dauer',
            'feature_18': 'prev_position_2_level',
            'feature_19': 'prev_position_2_branche',
            'feature_20': 'prev_position_2_dauer',

            # Company Size Feature
            'feature_21': 'company_size',

            # Study Field Feature
            'feature_22': 'study_field'
        }

        if len(df) > 0:
            feature_cols = [col for col in df.columns if col.startswith('feature_')]
            for i in range(min(5, len(feature_cols))):
                col = feature_cols[i]
                feature_name = feature_names.get(col, col)
                values = df[col].values
                print(f"  {feature_name} ({col}): min={values.min():.2f}, max={values.max():.2f}, mean={values.mean():.2f}")

        position_id_values = df['feature_8'].values
        unique_position_ids = set(position_id_values)
        print(f"  Position-ID values: {sorted(unique_position_ids)}")
        print(f"  Number of unique position IDs: {len(unique_position_ids)}")
        print(f"  Position-ID min: {position_id_values.min()}")
        print(f"  Position-ID max: {position_id_values.max()}")
        print(f"  Position-ID mean: {position_id_values.mean():.2f}") 

        position_id_mapping = {}
        for _, row in df.head(10).iterrows():
            position = row['position']
            position_id = row['feature_8']
            if position not in position_id_mapping:
                position_id_mapping[position] = position_id
                print(f"  '{position}' -> ID: {position_id}")

        if len(unique_position_ids) == 1 and 0 in unique_position_ids:
            logging.warning("Warning: All position IDs are 0!")
            logging.warning("Possible causes:")
            logging.warning("  - position_to_idx.json file is missing or empty")
            logging.warning("  - Positions are not correctly mapped")

        for i in range(min(10, 23)):
            feature_key = f'feature_{i}'
            feature_name = feature_names[feature_key]
            values = df[feature_key].values
            print(f"  {feature_name}: min={values.min():.2f}, max={values.max():.2f}, mean={values.mean():.2f}")

        def split_profile(group):
            n = len(group)
            split_idx = int(n * 0.7)
            group = group.sort_values("time_idx")
            group["is_train"] = [True]*split_idx + [False]*(n-split_idx)
            return group

        df = df.groupby("profile_id", group_keys=False).apply(split_profile)

        training = df[df["is_train"]].dropna()
        validation = df[~df["is_train"]].dropna()

        valid_profiles = set(training["profile_id"]).intersection(validation["profile_id"])
        self.training = training[training["profile_id"].isin(valid_profiles)]
        self.validation = validation[validation["profile_id"].isin(valid_profiles)]

        print("Train:", len(self.training), "Val:", len(self.validation))
        print(self.validation.head(20))

        unique_positions = df['position'].unique()
        position_to_id = {pos: i for i, pos in enumerate(unique_positions)}
        self.training['position_id'] = self.training['position'].map(position_to_id)
        self.validation['position_id'] = self.validation['position'].map(position_to_id)

        self.training['position_id'] = self.training['position_id'].astype(str)
        self.validation['position_id'] = self.validation['position_id'].astype(str)

        time_varying_unknown_reals = [f'feature_{i}' for i in range(23)]
        time_varying_known_reals = ['time_idx']
        static_categoricals = ['position_id']

        df_renamed = df.rename(columns=feature_names)

        print(f"Time-varying unknown reals: {len(time_varying_unknown_reals)} Features")
        print(f"Time-varying known reals: {time_varying_known_reals}")
        print(f"Static categoricals: {static_categoricals}")
        print(f"Anzahl eindeutige Positionen: {len(unique_positions)}")

        print(f"\nFeature-Übersicht:")
        for i in range(23):
            feature_key = f'feature_{i}'
            feature_name = feature_names[feature_key]
            print(f"  {feature_key} -> {feature_name}")

        self.training_renamed = self.training.rename(columns=feature_names)
        self.validation_renamed = self.validation.rename(columns=feature_names)

        time_varying_unknown_reals_named = [feature_names[f'feature_{i}'] for i in range(23)]

        '''
        Create TimeSeriesDataSet for training
        '''
        self.training_dataset = TimeSeriesDataSet(
            self.training_renamed,
            time_idx="time_idx",
            target="target",
            group_ids=["profile_id"],
            max_encoder_length=self.max_encoder_length,
            max_prediction_length=self.max_prediction_length,
            min_encoder_length=self.min_encoder_length,
            min_prediction_length=self.min_prediction_length,
            time_varying_unknown_reals=time_varying_unknown_reals_named,
            time_varying_known_reals=time_varying_known_reals,
            static_categoricals=static_categoricals,
            categorical_encoders={
                "profile_id": NaNLabelEncoder(add_nan=True),
                "position_id": NaNLabelEncoder(add_nan=True),
            }, 
            target_normalizer=GroupNormalizer(groups=["profile_id"], transformation="softplus"),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True,
        )

        '''
        Create TimeSeriesDataSet for validation
        '''
        self.validation_dataset = TimeSeriesDataSet.from_dataset(
            self.training_dataset,
            self.validation_renamed,
            predict=True,
            stop_randomization=True,
            allow_missing_timesteps=True,    
        )

    '''
    Train dataloader
    '''
    def train_dataloader(self):
        return self.training_dataset.to_dataloader(train=True, batch_size=self.batch_size, num_workers=4)

    '''
    Validation dataloader
    '''
    def val_dataloader(self):
        return self.validation_dataset.to_dataloader(train=False, batch_size=self.batch_size * 2, num_workers=4)