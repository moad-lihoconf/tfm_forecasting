import os
from contextlib import suppress
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn.functional as F
from pfns.bar_distribution import FullSupportBarDistribution
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder

from tfmplayground.model import NanoTabPFNModel
from tfmplayground.utils import get_default_device

_OFFICIAL_REGRESSOR_ARCH = {
    "num_attention_heads": 4,
    "embedding_size": 128,
    "mlp_hidden_size": 512,
    "num_layers": 6,
    "num_outputs": 100,
    "dropout": 0.0,
}

TargetNormalization = Literal["per_function_zscore", "per_function_clamped", "none"]


def _instantiate_model_from_arch_payload(arch: dict[str, Any]) -> NanoTabPFNModel:
    return NanoTabPFNModel(
        num_attention_heads=int(arch["num_attention_heads"]),
        embedding_size=int(arch["embedding_size"]),
        mlp_hidden_size=int(arch["mlp_hidden_size"]),
        num_layers=int(arch["num_layers"]),
        num_outputs=int(arch["num_outputs"]),
        dropout=float(arch.get("dropout", 0.0)),
        feature_normalization=str(
            arch.get("feature_normalization", "per_function_zscore")
        ),
        debug_output_clamp=(
            None
            if arch.get("debug_output_clamp") is None
            else float(arch["debug_output_clamp"])
        ),
    )


def _looks_like_raw_model_state_dict(state_dict: Any) -> bool:
    if not isinstance(state_dict, dict):
        return False
    required = {
        "feature_encoder.linear_layer.weight",
        "target_encoder.linear_layer.weight",
        "decoder.linear1.weight",
        "decoder.linear2.weight",
    }
    return required.issubset(state_dict.keys())


def _normalize_official_raw_state_dict_keys(
    state_dict: dict[str, Any],
) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in state_dict.items():
        new_key = key.replace(".self_attn_between_", ".self_attention_between_")
        normalized[new_key] = value
    return normalized


def init_model_from_state_dict_file(file_path):
    """
    Reads model architecture from state dict,
    instantiates the architecture and loads the weights.
    """
    model, _meta = _load_model_with_checkpoint_metadata(file_path)
    return model


def _extract_target_normalization_metadata(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """
    Extracts target normalization settings from known checkpoint payload shapes.

    Supported shapes:
    - train.py "latest_checkpoint.pth"/"best_checkpoint.pth": top-level keys
      "target_normalization" and "target_std_floor".
    - pretrain_regression.py best-weights payload: nested under "training".
    """

    training = payload.get("training")
    if isinstance(training, dict):
        target_normalization = training.get("target_normalization")
        target_std_floor = training.get("target_std_floor")
        if target_normalization is not None:
            return {
                "target_normalization": target_normalization,
                "target_std_floor": target_std_floor,
            }

    # Legacy / training_state payload.
    if "target_normalization" in payload:
        return {
            "target_normalization": payload.get("target_normalization"),
            "target_std_floor": payload.get("target_std_floor"),
        }

    return {}


def _load_model_with_checkpoint_metadata(
    file_path: str,
) -> tuple[NanoTabPFNModel, dict[str, Any]]:
    """
    Loads a NanoTabPFNModel from a checkpoint file and returns extracted
    metadata that may influence inference behavior.
    """
    payload = torch.load(file_path, map_location=torch.device("cpu"))

    if isinstance(payload, dict) and "architecture" in payload and "model" in payload:
        model = _instantiate_model_from_arch_payload(
            cast(dict[str, Any], payload["architecture"])
        )
        model.load_state_dict(cast(dict[str, Any], payload["model"]))
        meta = _extract_target_normalization_metadata(cast(dict[str, Any], payload))
        return model, meta

    if _looks_like_raw_model_state_dict(payload):
        model = _instantiate_model_from_arch_payload(_OFFICIAL_REGRESSOR_ARCH)
        model.load_state_dict(
            _normalize_official_raw_state_dict_keys(cast(dict[str, Any], payload))
        )
        return model, {}

    raise ValueError(
        "Unsupported checkpoint format in "
        f"{file_path!r}: expected wrapped checkpoint with architecture/model "
        "or the official raw standard-regressor state dict."
    )


# doing these as lambdas would cause NanoTabPFNClassifier to not be pickle-able,
# which would cause issues if we want to run it inside the tabarena codebase
def to_pandas(x):
    return pd.DataFrame(x) if not isinstance(x, pd.DataFrame) else x


def to_numeric(x):
    return x.apply(pd.to_numeric, errors="coerce").to_numpy()


def get_feature_preprocessor(X: np.ndarray | pd.DataFrame) -> ColumnTransformer:
    """
    Fits a preprocessor that imputes NaNs, encodes
    categorical features and removes constant features.
    """
    X = pd.DataFrame(X)
    num_mask: list[bool] = []
    cat_mask: list[bool] = []
    for col in X:
        unique_non_nan_entries = X[col].dropna().unique()
        if len(unique_non_nan_entries) <= 1:
            num_mask.append(False)
            cat_mask.append(False)
            continue
        non_nan_entries = X[col].notna().sum()
        numeric_entries = (
            pd.to_numeric(X[col], errors="coerce").notna().sum()
        )  # in case numeric columns are stored as strings
        num_mask.append(non_nan_entries == numeric_entries)
        cat_mask.append(non_nan_entries != numeric_entries)
        # num_mask.append(is_numeric_dtype(X[col]))  # Assumes pandas dtype is correct

    num_mask_arr = np.array(num_mask)
    cat_mask_arr = np.array(cat_mask)

    num_transformer = Pipeline(
        [
            (
                "to_pandas",
                FunctionTransformer(to_pandas),
            ),  # to apply pd.to_numeric of pandas
            (
                "to_numeric",
                FunctionTransformer(to_numeric),
            ),  # in case numeric columns are stored as strings
            (
                "imputer",
                SimpleImputer(strategy="mean", add_indicator=True),
            ),  # median might be better because of outliers
        ]
    )
    cat_transformer = Pipeline(
        [
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=np.nan
                ),
            ),
            ("imputer", SimpleImputer(strategy="most_frequent", add_indicator=True)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_mask_arr),
            ("cat", cat_transformer, cat_mask_arr),
        ]
    )
    return preprocessor


class NanoTabPFNClassifier:
    """scikit-learn like interface"""

    def __init__(
        self,
        model: NanoTabPFNModel | str | None = None,
        device: None | str | torch.device = None,
        num_mem_chunks: int = 8,
    ):
        if device is None:
            device = get_default_device()
        if model is None:
            model = "checkpoints/nanotabpfn.pth"
            if not os.path.isfile(model):
                os.makedirs("checkpoints", exist_ok=True)
                print("No cached model found, downloading model checkpoint.")
                response = requests.get(
                    "https://ml.informatik.uni-freiburg.de/research-artifacts/pfefferle/TFM-Playground/nanotabpfn_classifier.pth"
                )
                with open(model, "wb") as f:
                    f.write(response.content)
        if isinstance(model, str):
            model = init_model_from_state_dict_file(model)
        assert isinstance(model, NanoTabPFNModel)
        self.model = model.to(device)
        self.device = device
        self.num_mem_chunks = num_mem_chunks

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Stores X_train and y_train for later use, also computes num_classes."""
        self.feature_preprocessor = get_feature_preprocessor(X_train)
        self.X_train = self.feature_preprocessor.fit_transform(X_train)
        self.y_train = y_train
        self.num_classes = max(set(y_train)) + 1

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Calls predict_proba and picks the class with the highest probability."""
        predicted_probabilities = self.predict_proba(X_test)
        return predicted_probabilities.argmax(axis=1)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        creates (x,y), runs it through our PyTorch Model,
        cuts off the classes that didn't appear in the training data
        and applies softmax to get the probabilities
        """
        x_np = np.concatenate(
            (self.X_train, self.feature_preprocessor.transform(X_test))
        )
        with torch.no_grad():
            x = (
                torch.from_numpy(x_np).unsqueeze(0).to(torch.float).to(self.device)
            )  # introduce batch size 1
            y = (
                torch.from_numpy(self.y_train)
                .unsqueeze(0)
                .to(torch.float)
                .to(self.device)
            )
            out = self.model(
                (x, y),
                single_eval_pos=len(self.X_train),
                num_mem_chunks=self.num_mem_chunks,
            ).squeeze(0)  # remove batch size 1
            # our pretrained classifier supports up to num_outputs classes,
            # if the dataset has less we cut off the rest
            out = out[:, : self.num_classes]
            # apply softmax to get a probability distribution
            probabilities = F.softmax(out, dim=1)
            return probabilities.to("cpu").numpy()


class NanoTabPFNRegressor:
    """scikit-learn like interface"""

    def __init__(
        self,
        model: NanoTabPFNModel | str | None = None,
        dist: FullSupportBarDistribution | str | None = None,
        device: str | torch.device | None = None,
        num_mem_chunks: int = 8,
        target_normalization: TargetNormalization | None = None,
        target_std_floor: float | None = None,
    ):
        if device is None:
            device = get_default_device()
        self.target_normalization: TargetNormalization | None = target_normalization
        self.target_std_floor: float = (
            1e-2 if target_std_floor is None else float(target_std_floor)
        )
        if model is None:
            os.makedirs("checkpoints", exist_ok=True)
            model = "checkpoints/nanotabpfn_regressor.pth"
            dist = "checkpoints/nanotabpfn_regressor_buckets.pth"
            if not os.path.isfile(model):
                print("No cached model found, downloading model checkpoint.")
                response = requests.get(
                    "https://ml.informatik.uni-freiburg.de/research-artifacts/pfefferle/TFM-Playground/nanotabpfn_regressor.pth"
                )
                with open(model, "wb") as f:
                    f.write(response.content)
            if not os.path.isfile(dist):
                print("No cached bucket edges found, downloading bucket edges.")
                response = requests.get(
                    "https://ml.informatik.uni-freiburg.de/research-artifacts/pfefferle/TFM-Playground/nanotabpfn_regressor_buckets.pth"
                )
                with open(dist, "wb") as f:
                    f.write(response.content)
        if isinstance(model, str):
            model, meta = _load_model_with_checkpoint_metadata(model)
            if self.target_normalization is None:
                maybe_norm = meta.get("target_normalization")
                if isinstance(maybe_norm, str):
                    self.target_normalization = cast(TargetNormalization, maybe_norm)
            if target_std_floor is None:
                maybe_floor = meta.get("target_std_floor")
                if maybe_floor is not None:
                    with suppress(TypeError, ValueError):
                        self.target_std_floor = float(maybe_floor)

        if isinstance(dist, str):
            dist_payload = torch.load(dist, map_location=device)
            if isinstance(dist_payload, dict):
                bucket_edges = dist_payload.get("bucket_edges")
                dist = (
                    FullSupportBarDistribution(bucket_edges).float()
                    if bucket_edges is not None
                    else None
                )
            else:
                dist = FullSupportBarDistribution(dist_payload).float()

        assert isinstance(model, NanoTabPFNModel)
        self.model = model.to(device)
        self.device = device
        self.dist = dist
        self.num_mem_chunks = num_mem_chunks
        if self.target_normalization is None:
            # Default behavior matches the official released regressor wrapper.
            self.target_normalization = "per_function_zscore"
        if self.target_normalization not in {
            "per_function_zscore",
            "per_function_clamped",
            "none",
        }:
            raise ValueError(
                f"Unsupported target_normalization: {self.target_normalization!r}"
            )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Stores X_train and y_train for later use.
        Computes target normalization.
        """
        self.feature_preprocessor = get_feature_preprocessor(X_train)
        self.X_train = self.feature_preprocessor.fit_transform(X_train)
        self.y_train = y_train

        if self.target_normalization == "none":
            self.y_train_mean = 0.0
            self.y_train_std = 1.0
            self.y_train_n = self.y_train
            return

        self.y_train_mean = float(np.mean(self.y_train))
        raw_std = float(np.std(self.y_train, ddof=1))
        if not np.isfinite(raw_std):
            raw_std = 0.0
        if self.target_normalization == "per_function_clamped":
            raw_std = max(raw_std, float(self.target_std_floor))
        self.y_train_std = raw_std + 1e-8
        self.y_train_n = (self.y_train - self.y_train_mean) / self.y_train_std

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Performs in-context learning using X_train and y_train.
        Predicts the means of the output distributions for X_test.
        Renormalizes the predictions back to the original target scale.
        """
        X = np.concatenate((self.X_train, self.feature_preprocessor.transform(X_test)))
        y = self.y_train_n

        with torch.no_grad():
            X_tensor = torch.tensor(
                X, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            y_tensor = torch.tensor(
                y, dtype=torch.float32, device=self.device
            ).unsqueeze(0)

            logits = self.model(
                (X_tensor, y_tensor),
                single_eval_pos=len(self.X_train),
                num_mem_chunks=self.num_mem_chunks,
            ).squeeze(0)
            if isinstance(self.dist, FullSupportBarDistribution):
                preds_n = self.dist.mean(logits)
            else:
                if logits.ndim == 2 and logits.shape[-1] == 1:
                    preds_n = logits.squeeze(-1)
                elif logits.ndim == 1:
                    preds_n = logits
                else:
                    raise ValueError(
                        "Scalar regression checkpoints require a single-output "
                        "model or a bar-distribution artifact."
                    )
            preds = preds_n * self.y_train_std + self.y_train_mean

        return preds.cpu().numpy()
