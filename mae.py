#!/usr/bin/env python3
"""
MAE-style ECG pretraining + finetuning script using the exact same preprocessing
pipeline as TinyTransformer2402.

Pipeline:
1) Extract 198-sample beats and RR intervals from MIT-BIH
2) Optional denoising
3) Row-wise normalization
4) 7:1:2 train/valid/test split
5) MAE pretraining on train beats only
6) Finetune classifier initialized from MAE encoder
7) Report accuracy / balanced_accuracy / macro_f1

MAE design:
- Same Conv1d tokenizer as TinyTransformer2402
- Same encoder dimension: 16
- Same encoder depth: 1 block
- Same token count: 56
- Random masking over tokens
- Lightweight decoder reconstructs masked token embeddings
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainerCallback, TrainingArguments
import matplotlib.pyplot as plt

from common.lib import (
    SEED,
    FS,
    WINDOW,
    IDX2CLS,
    seed_everything,
    maybe_augment_noise,
    extract_beats_and_rr,
    preprocess_beats,
    balance_classes,
    ECGRRDataset,
    compute_metrics,
    percent_trained,
    make_training_args,
)
from novel.mae_lib import (
    ECGMAEDataset,
    mae_collator,
    cls_collator,
    add_common_ecg_cli_args,
)

class MAEReconstructionCallback(TrainerCallback):

    def __init__(self, val_dataset, model, save_dir, num_samples=6, interval=10):
        """
        val_dataset : validation dataset
        model       : MAE model
        save_dir    : directory for reconstruction plots
        num_samples : number of signals to visualize
        interval    : save every N epochs
        """

        self.val_dataset = val_dataset
        self.model = model
        self.save_dir = save_dir
        self.num_samples = num_samples
        self.interval = interval

        os.makedirs(save_dir, exist_ok=True)

    def unpatchify(self, patches, patch_size):
        """
        patches: [B, N, patch_size]
        returns: [B, 1, seq_len]
        """
        B, N, P = patches.shape
        seq_len = N * P
        x = patches.reshape(B, seq_len)
        return x.unsqueeze(1)

    def reconstruct_signal(self, x):

        self.model.eval()

        with torch.no_grad():

            latent, target, mask, ids_restore = self.model.forward_encoder(x)

            pred = self.model.forward_decoder(latent, ids_restore)

            recon = self.unpatchify(pred, self.model.patch_size)

        return x, recon, mask

    def plot_reconstruction(self, original, recon, mask, save_path):

        patch_size = self.model.patch_size

        mask = mask.cpu().numpy()
        original = original.cpu().numpy()
        recon = recon.cpu().numpy()

        B = original.shape[0]

        fig, axes = plt.subplots(B, 1, figsize=(10, 2 * B))

        if B == 1:
            axes = [axes]

        for i in range(B):

            axes[i].plot(original[i, 0], label="original", linewidth=1.5)
            axes[i].plot(recon[i, 0], label="reconstruction", linewidth=1)

            for j, m in enumerate(mask[i]):

                if m == 1:

                    start = j * patch_size
                    end = start + patch_size

                    axes[i].axvspan(start, end, color="red", alpha=0.1)

            axes[i].set_xlim(0, original.shape[-1])
            axes[i].set_yticks([])

        axes[0].legend()

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def on_epoch_end(self, args, state, control, **kwargs):

        epoch = int(state.epoch)

        if epoch % self.interval != 0:
            return

        idx = np.random.choice(len(self.val_dataset), self.num_samples, replace=False)

        xs = [self.val_dataset[i]["x"] for i in idx]

        x = torch.stack(xs).to(self.model.pos.device)

        original, recon, mask = self.reconstruct_signal(x)

        save_path = os.path.join(self.save_dir, f"epoch_{epoch}.png")

        self.plot_reconstruction(original, recon, mask, save_path)

def make_training_args(
    output_dir: str,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    save_strategy: str = "epoch",
) -> TrainingArguments:
    kwargs = dict(
        output_dir=output_dir,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        save_strategy=save_strategy,
        logging_strategy="epoch",
        load_best_model_at_end=True,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=seed,
        save_total_limit=2,
        remove_unused_columns=False,
    )

    try:
        return TrainingArguments(eval_strategy="epoch", **kwargs)
    except TypeError:
        return TrainingArguments(evaluation_strategy="epoch", **kwargs)

class ECGMAE(nn.Module):

    def __init__(
        self,
        seq_len=198,
        patch_size=9,
        embed_dim=64,
        decoder_dim=32,
        nhead=8,
        mask_ratio=0.6,
        n_layer=4
    ):
        super().__init__()

        self.seq_len = seq_len
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.num_patches = seq_len // patch_size
        self.embed_dim = embed_dim
        self.n_layer = n_layer

        # patch embedding (like ViT)
        self.patch_embed = nn.Conv1d(
            1,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        self.pos = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        enc = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=4*embed_dim,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )

        self.encoder = nn.TransformerEncoder(enc, num_layers=n_layer)
        self.encoder_norm = nn.LayerNorm(embed_dim)

        # decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_dim)

        self.mask_token = nn.Parameter(torch.zeros(1,1,decoder_dim))
        self.decoder_pos = nn.Parameter(torch.zeros(1,self.num_patches,decoder_dim))

        dec = nn.TransformerEncoderLayer(
            d_model=decoder_dim,
            nhead=4,
            dim_feedforward=decoder_dim*4,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )

        self.decoder = nn.TransformerEncoder(dec, num_layers=1)
        self.decoder_norm = nn.LayerNorm(decoder_dim)

        # predict RAW PATCH VALUES
        self.decoder_pred = nn.Linear(decoder_dim, patch_size)

        self.loss_fn = nn.MSELoss(reduction="none")

    def patchify(self, x):

        patches = x.unfold(2, self.patch_size, self.patch_size)

        return patches.squeeze(1)

    def tokenize(self, x):

        tokens = self.patch_embed(x).transpose(1,2)

        return tokens

    def random_masking(self, x):

        B,N,D = x.shape

        len_keep = int(N*(1-self.mask_ratio))

        noise = torch.rand(B,N,device=x.device)

        ids_shuffle = torch.argsort(noise,dim=1)

        ids_restore = torch.argsort(ids_shuffle,dim=1)

        ids_keep = ids_shuffle[:,:len_keep]

        x_masked = torch.gather(
            x,
            1,
            ids_keep.unsqueeze(-1).expand(-1,-1,D)
        )

        mask = torch.ones(B,N,device=x.device)

        mask[:,:len_keep] = 0

        mask = torch.gather(mask,1,ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self,x):
        target = self.patchify(x).detach()
        tokens = self.tokenize(x)
        tokens = tokens + self.pos
        x_masked,mask,ids_restore = self.random_masking(tokens)
        latent = self.encoder(x_masked)
        latent = self.encoder_norm(latent)
        return latent,target,mask,ids_restore

    def forward_decoder(self,latent,ids_restore):
        x = self.decoder_embed(latent)
        B,N_keep,D = x.shape
        N = ids_restore.shape[1]
        mask_tokens = self.mask_token.repeat(B,N-N_keep,1)
        x_ = torch.cat([x,mask_tokens],dim=1)
        x_ = torch.gather(
            x_,
            1,
            ids_restore.unsqueeze(-1).expand(-1,-1,D)
        )
        x_ = x_ + self.decoder_pos
        x_ = self.decoder(x_)
        x_ = self.decoder_norm(x_)
        pred = self.decoder_pred(x_)
        return pred

    def forward_loss(self,target,pred,mask):

        loss = self.loss_fn(pred,target).mean(dim=-1)

        loss = (loss*mask).sum()/mask.sum()

        return loss

    def forward(self,x=None,labels=None):

        latent,target,mask,ids_restore = self.forward_encoder(x)

        pred = self.forward_decoder(latent,ids_restore)

        loss = self.forward_loss(target,pred,mask)

        return {"loss":loss,"logits":pred}
    def build_classifier(self, n_classes: int = 5, class_weights: torch.Tensor | None = None):
        model = TinyTransformerMAEFinetune(
            n_classes=n_classes,
            patch_size=self.patch_size,
            seq_len=self.seq_len,
            embed_dim=self.embed_dim,
            n_layer=self.n_layer,
            class_weights=class_weights,
        )

        model.patch_embed.load_state_dict(self.patch_embed.state_dict())

        model.pos.data.copy_(self.pos.data)

        model.encoder.load_state_dict(self.encoder.state_dict())

        model.final_norm.load_state_dict(self.encoder_norm.state_dict())

        return model

class TinyTransformerMAEFinetune(nn.Module):

    def __init__(
        self,
        n_layer=1,
        embed_dim=128,
        n_classes: int = 5,
        patch_size: int = 9,
        seq_len: int = 198,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()

        num_patches = seq_len // patch_size

        # SAME tokenizer as MAE
        self.patch_embed = nn.Conv1d(
            1,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.pos = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        enc = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=4*embed_dim,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            dropout=0.2,
        )

        self.encoder = nn.TransformerEncoder(enc, num_layers=n_layer)
        self.final_norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)

        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, x=None, labels=None):
        x_tokens = self.patch_embed(x).transpose(1, 2)
        x_tokens = x_tokens + self.pos[:, :x_tokens.size(1), :]

        x_tokens = self.encoder(x_tokens)

        x_tokens = self.final_norm(x_tokens)

        x_pooled = x_tokens.mean(dim=1)
        logits = self.head(x_pooled)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}

    def get_attention_map(self, x: torch.Tensor, layer_idx: int = 0) -> torch.Tensor:
        """
        Compute the self-attention map for a given input batch on a specific encoder layer.

        Returns attention weights averaged over heads with shape [B, N, N],
        where N is the number of tokens (patches).
        """
        # Tokenize + add positional encodings (same as in forward)
        x_tokens = self.patch_embed(x).transpose(1, 2)
        x_tokens = x_tokens + self.pos[:, :x_tokens.size(1), :]

        # Select encoder layer
        if layer_idx < 0 or layer_idx >= len(self.encoder.layers):
            raise ValueError(f"layer_idx out of range: {layer_idx}")

        layer = self.encoder.layers[layer_idx]

        # Run only the self-attention of this layer with need_weights=True
        # to obtain the attention map. We skip the feed-forward part here,
        # since we're interested only in the attention weights.
        attn_output, attn_weights = layer.self_attn(
            x_tokens,
            x_tokens,
            x_tokens,
            need_weights=True,
        )

        # attn_weights: [B, num_heads, N, N] -> average over heads
        attn_mean = attn_weights.mean(dim=1)
        return attn_mean


def plot_attention_map(attn: torch.Tensor, save_path: str | None = None) -> None:
    """
    Plot a single attention map (for one example) as a heatmap.

    Expects attn of shape [N, N] or [1, N, N]. If batched, uses the first item.
    """
    if attn.dim() == 3:
        attn = attn[0]

    attn_np = attn.detach().cpu().numpy()

    plt.figure(figsize=(6, 5))
    plt.imshow(attn_np, cmap="viridis", aspect="auto")
    plt.colorbar(label="Attention weight")
    plt.xlabel("Key token index")
    plt.ylabel("Query token index")
    plt.title("Self-attention map")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    add_common_ecg_cli_args(parser, output_dir_default="./tiny_ecg_mae_runs")

    parser.add_argument("--mask_ratio", type=float, default=0.6)
    args = parser.parse_args()

    seed_everything(SEED)

    X, RR, y = extract_beats_and_rr(args.folder, pre_process=None)
    X = preprocess_beats(X)

    print(f"Loaded beats: {len(y)}")
    class_counts = {IDX2CLS[i]: int((y == i).sum()) for i in range(5)}
    print("Class counts:", class_counts)

    # Same 7:1:2 split as TinyTransformer
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_tmp, y_tmp, test_size=2 / 3, stratify=y_tmp, random_state=SEED
    )

    if args.use_noise_aug:
        X_train = maybe_augment_noise(X_train, args.nstdb_folder, args.snr_db)

    mae_train_dataset = ECGMAEDataset(X_train)
    mae_valid_dataset = ECGMAEDataset(X_valid)

    print("\n=== Stage 1: MAE pretraining ===")
    mae_model = ECGMAE(mask_ratio=args.mask_ratio)

    mae_args = make_training_args(
        output_dir=str(Path(args.output_dir) / "mae_pretrain"),
        epochs=args.pretrain_epochs,
        batch_size=args.pretrain_batch_size,
        lr=args.pretrain_lr,
        seed=SEED,
    )
    mae_args.metric_for_best_model = "eval_loss"
    mae_args.greater_is_better = False

    if args.checkpoint == None:
        mae_trainer = Trainer(
            model=mae_model,
            args=mae_args,
            train_dataset=mae_train_dataset,
            eval_dataset=mae_valid_dataset,
            data_collator=mae_collator,
            callbacks=[
            MAEReconstructionCallback(
                val_dataset=mae_valid_dataset,
                model=mae_model,
                save_dir="./tiny_ecg_mae_runs/reconstruction"
            )
        ],
        )
        mae_trainer.train()
        print("MAE validation:")
        print(mae_trainer.evaluate())
    else:
        from safetensors.torch import load_file
        state_dict = load_file(args.checkpoint)
        mae_model.load_state_dict(state_dict)

    print("\n=== Stage 2: classifier finetuning ===")

    X_train, y_train = percent_trained(X_train, y_train, args)

    # Rebalance only the training set
    X_train, y_train = balance_classes(
        X_train,
        y_train,
        target_size=5000,
        seed=SEED,
        n_classes=5,
    )

    class_counts = np.bincount(y_train, minlength=5).astype(np.float32)
    class_weights_np = class_counts.sum() / (len(class_counts) * class_counts + 1e-8)
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32)

    clf_model = mae_model.build_classifier(
        n_classes=len(class_counts),
        class_weights=class_weights if args.balanced_weight else None,
    )

    train_dataset = ECGRRDataset(X_train, y_train)
    valid_dataset = ECGRRDataset(X_valid, y_valid)
    test_dataset = ECGRRDataset(X_test, y_test)

    finetune_args = make_training_args(
        output_dir=str(Path(args.output_dir) / "finetune"),
        epochs=args.finetune_epochs,
        batch_size=args.finetune_batch_size,
        lr=args.finetune_lr,
        seed=SEED,
    )
    finetune_args.metric_for_best_model = "macro_f1"
    finetune_args.greater_is_better = True

    clf_trainer = Trainer(
        model=clf_model,
        args=finetune_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=cls_collator,
        compute_metrics=compute_metrics,
    )
    clf_trainer.train()

    print("Validation metrics:")
    val_metrics = clf_trainer.evaluate()
    print(val_metrics)

    print("Test metrics:")
    test_metrics = clf_trainer.evaluate(eval_dataset=test_dataset)
    print(test_metrics)

    pred_output = clf_trainer.predict(test_dataset)
    y_pred = np.argmax(pred_output.predictions, axis=1)
    y_true = pred_output.label_ids

    print(classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2, 3, 4],
        target_names=[IDX2CLS[i] for i in range(5)],
        zero_division=0,
    ))
    print(confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1, 2, 3, 4],
    ))


if __name__ == "__main__":
    main()