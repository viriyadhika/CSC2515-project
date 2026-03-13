from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.model_selection import train_test_split

from audio.utils import ESC50 as ESC50Loader
from common.lib import (
    SEED,
    IDX2CLS,
    extract_beats_and_rr,
    preprocess_beats,
    maybe_augment_noise,
)


ESC10_TARGETS = [0, 1, 10, 11, 12, 20, 21, 38, 40, 41]
ESC50_CSV_PATH = "data/esc50.csv"
ESC50_WAV_DIR = "data/audio/audio"
ESC50_AUDIO_RATE = 8000
ESC50_INPUT_LENGTH = 5.0
ESC50_VALID_FOLD = 4
ESC50_TEST_FOLD = 5


@dataclass
class ECGLoader:
    args: object
    pre_process: Callable[[np.ndarray], np.ndarray] | None = None
    post_process: Callable[[np.ndarray], np.ndarray] | None = preprocess_beats
    include_rr: bool = False

    def load_full(self) -> dict[str, object]:
        X, RR, y = extract_beats_and_rr(self.args.folder, pre_process=self.pre_process)
        if self.post_process is not None:
            X = self.post_process(X)

        print(f"Loaded beats: {len(y)}")
        class_counts = {IDX2CLS[i]: int((y == i).sum()) for i in range(5)}
        print("Class counts:", class_counts)

        data = {
            "X": X.astype(np.float32),
            "y": y.astype(np.int64),
            "label_names": [IDX2CLS[i] for i in range(5)],
            "seq_len": int(getattr(self.args, "seq_len", None) or X.shape[1]),
            "patch_size": int(getattr(self.args, "patch_size", None) or 9),
            "n_classes": 5,
            "dataset_name": "ecg",
        }
        if self.include_rr:
            data["RR"] = RR.astype(np.float32)
        return data

    def load(self) -> dict[str, object]:
        full = self.load_full()
        X = full["X"]
        y = full["y"]
        RR = full.get("RR")

        if RR is None:
            X_train, X_tmp, y_train, y_tmp = train_test_split(
                X, y, test_size=0.30, stratify=y, random_state=SEED
            )
            X_valid, X_test, y_valid, y_test = train_test_split(
                X_tmp, y_tmp, test_size=2 / 3, stratify=y_tmp, random_state=SEED
            )
            if getattr(self.args, "use_noise_aug", False):
                X_train = maybe_augment_noise(
                    X_train, getattr(self.args, "nstdb_folder", None), self.args.snr_db
                )
            return {
                "X_train": X_train.astype(np.float32),
                "X_valid": X_valid.astype(np.float32),
                "X_test": X_test.astype(np.float32),
                "y_train": y_train.astype(np.int64),
                "y_valid": y_valid.astype(np.int64),
                "y_test": y_test.astype(np.int64),
                "label_names": full["label_names"],
                "seq_len": full["seq_len"],
                "patch_size": full["patch_size"],
                "n_classes": full["n_classes"],
                "dataset_name": full["dataset_name"],
            }

        X_train, X_tmp, RR_train, RR_tmp, y_train, y_tmp = train_test_split(
            X, RR, y, test_size=0.30, stratify=y, random_state=SEED
        )
        X_valid, X_test, RR_valid, RR_test, y_valid, y_test = train_test_split(
            X_tmp, RR_tmp, y_tmp, test_size=2 / 3, stratify=y_tmp, random_state=SEED
        )
        if getattr(self.args, "use_noise_aug", False):
            X_train = maybe_augment_noise(
                X_train, getattr(self.args, "nstdb_folder", None), self.args.snr_db
            )
        return {
            "X_train": X_train.astype(np.float32),
            "X_valid": X_valid.astype(np.float32),
            "X_test": X_test.astype(np.float32),
            "RR_train": RR_train.astype(np.float32),
            "RR_valid": RR_valid.astype(np.float32),
            "RR_test": RR_test.astype(np.float32),
            "y_train": y_train.astype(np.int64),
            "y_valid": y_valid.astype(np.int64),
            "y_test": y_test.astype(np.int64),
            "label_names": full["label_names"],
            "seq_len": full["seq_len"],
            "patch_size": full["patch_size"],
            "n_classes": full["n_classes"],
            "dataset_name": full["dataset_name"],
        }


@dataclass
class AudioLoader:
    args: object
    only_esc10: bool = False

    def load(self) -> dict[str, object]:
        all_folds = {1, 2, 3, 4, 5}
        train_folds = sorted(all_folds - {ESC50_VALID_FOLD, ESC50_TEST_FOLD})

        base_loader = ESC50Loader(
            csv_path=ESC50_CSV_PATH,
            wav_dir=ESC50_WAV_DIR,
            only_ESC10=self.only_esc10,
            folds=sorted(all_folds),
            randomize=False,
            audio_rate=ESC50_AUDIO_RATE,
            strongAugment=False,
            pad=0,
            inputLength=ESC50_INPUT_LENGTH,
            random_crop=False,
            mix=False,
            normalize=True,
        )
        if self.only_esc10:
            base_loader.df = base_loader.df[base_loader.df["esc10"]].copy()

        label_to_idx, label_names = self._build_label_map(base_loader.df)
        X_train, y_train = self._materialize_split(base_loader, train_folds, label_to_idx)
        X_valid, y_valid = self._materialize_split(base_loader, [ESC50_VALID_FOLD], label_to_idx)
        X_test, y_test = self._materialize_split(base_loader, [ESC50_TEST_FOLD], label_to_idx)

        class_counts = {
            label_names[i]: int((y_train == i).sum() + (y_valid == i).sum() + (y_test == i).sum())
            for i in range(len(label_names))
        }
        print(
            f"Loaded ESC-50 clips: train={len(y_train)} valid={len(y_valid)} test={len(y_test)}"
        )
        print("Class counts:", class_counts)

        return {
            "X_train": X_train.astype(np.float32),
            "X_valid": X_valid.astype(np.float32),
            "X_test": X_test.astype(np.float32),
            "y_train": y_train.astype(np.int64),
            "y_valid": y_valid.astype(np.int64),
            "y_test": y_test.astype(np.int64),
            "label_names": label_names,
            "seq_len": int(getattr(self.args, "seq_len", None) or X_train.shape[1]),
            "patch_size": int(getattr(self.args, "patch_size", None) or 400),
            "n_classes": len(label_names),
            "dataset_name": "esc50",
        }

    def _build_label_map(self, df) -> tuple[dict[int, int], list[str]]:
        if self.only_esc10:
            target_order = ESC10_TARGETS
        else:
            target_order = sorted(int(t) for t in df["target"].unique())

        label_to_idx = {target: idx for idx, target in enumerate(target_order)}
        label_names = []
        for target in target_order:
            category = df.loc[df["target"] == target, "category"].iloc[0]
            label_names.append(str(category))
        return label_to_idx, label_names

    def _materialize_split(
        self,
        loader: ESC50Loader,
        folds: list[int],
        label_to_idx: dict[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        split_df = loader.df[loader.df["fold"].isin(folds)].reset_index(drop=True)
        if split_df.empty:
            raise ValueError(f"No ESC-50 samples found for folds={folds}")

        sounds = []
        labels = []
        for row in split_df.itertuples(index=False):
            sound = loader.fname_to_wav(row.filename)
            sound = loader.preprocess(sound).astype(np.float32)
            sounds.append(sound)
            labels.append(label_to_idx[int(row.target)])

        try:
            X = np.stack(sounds).astype(np.float32)
        except ValueError as exc:
            raise ValueError(
                "ESC-50 preprocessing produced variable-length audio. "
                "Use a fixed audio rate / crop setup before stacking."
            ) from exc

        y = np.asarray(labels, dtype=np.int64)
        return X, y
