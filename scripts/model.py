import torch
import torch.nn as nn
import torch.optim as torch_optim
from lightning.pytorch import LightningModule
from torchvision import models
import torch_optimizer as optim
import os
import csv
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, recall_score, classification_report, confusion_matrix


CLASS_NAMES = [
    "Arable land",
    "Permanent crops",
    "Grassland",
    "Wooded areas",
    "Shrubs",
    "Bare surface, low or rare vegetation",
    "Artificial constructions and sealed areas",
    "Inland waters / Transitional waters and Coastal waters",
]


class LULC_Model(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        self.network = models.wide_resnet101_2(weights=models.Wide_ResNet101_2_Weights.IMAGENET1K_V2)
        n_inputs = self.network.fc.in_features
        self.backbone_frozen = False
        self.network.fc = nn.Sequential(
            nn.Linear(n_inputs, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
        self.aef_lambda = torch.nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.aef_head = torch.nn.Linear(64, num_classes)
        nn.init.zeros_(self.aef_head.weight)
        nn.init.zeros_(self.aef_head.bias)

    def forward(self, xb, embeddings=None):
        image_logits = self.network(xb)

        if embeddings is None:
            return image_logits

        embeddings = embeddings.float()
        if embeddings.ndim > 2:
            embeddings = embeddings.flatten(start_dim=1)
        aef_logits = self.aef_head(embeddings)
        return image_logits + self.aef_lambda * aef_logits

    def freeze(self):
        self.network.requires_grad_(False)
        self.aef_head.requires_grad_(True)
        self.aef_lambda.requires_grad = True
        self.backbone_frozen = True
        self.network.eval()

    def unfreeze(self):
        self.network.requires_grad_(True)
        self.backbone_frozen = False
        if self.training:
            self.network.train()

    def train(self, mode=True):
        super().train(mode)
        if mode and self.backbone_frozen:
            self.network.eval()
        return self


class LitClassifier(LightningModule):
    def __init__(self, num_classes=8, lr=1e-3, freeze_resnet=False, warmup_epochs=0):
        super().__init__()
        self.save_hyperparameters()
        self.model = LULC_Model(num_classes=num_classes)
        if freeze_resnet:
            self.model.freeze()
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.best_val_oa = 0.0
        self.save_dir = None
        self.val_outputs = []
        self.warmup_epochs = warmup_epochs

    def forward(self, x, embeddings=None):
        return self.model(x, embeddings=embeddings)

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        embedding = batch.get("embedding")
        logits = self(x, embeddings=embedding)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, batch_size=x.size(0))
        self.log("train_acc", acc, batch_size=x.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        embedding = batch.get("embedding")
        logits = self(x, embeddings=embedding)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_acc", acc, prog_bar=True, batch_size=x.size(0))
        self.log("val_loss", loss, prog_bar=True, batch_size=x.size(0))
        self.val_outputs.append({"logits": logits.detach().cpu(), "labels": y.detach().cpu()})
        return None
    
    def test_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        embedding = batch.get("embedding")
        logits = self(x, embeddings=embedding)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("test_loss", loss, prog_bar=True, batch_size=x.size(0))
        self.log("test_acc", acc, prog_bar=True, batch_size=x.size(0))
        return {"test_loss": loss, "test_acc": acc, "probabilities": logits.softmax(dim=1)}

    def on_validation_epoch_start(self):
        self.val_outputs.clear()

    def on_validation_epoch_end(self):
        """Calculate and log validation metrics at the end of each epoch."""
        if not self.val_outputs:
            return
        logits = torch.cat([o["logits"] for o in self.val_outputs], dim=0)
        labels = torch.cat([o["labels"] for o in self.val_outputs], dim=0)
        preds = logits.argmax(dim=1)
        all_preds = preds.numpy()
        all_targets = labels.numpy()
        oa = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
        aa = recall_score(all_targets, all_preds, average="macro", zero_division=0)
        self.log("val_OverallAccuracy", oa, prog_bar=False, on_epoch=True)
        self.log("val_F1Score", f1, prog_bar=False, on_epoch=True)
        self.log("val_AverageAccuracy", aa, prog_bar=False, on_epoch=True)
        if oa > self.best_val_oa:
            self.best_val_oa = oa

    def _collect_validation_outputs(self):
        logits = torch.cat([o["logits"] for o in self.val_outputs], dim=0)
        labels = torch.cat([o["labels"] for o in self.val_outputs], dim=0)
        preds = logits.argmax(dim=1)
        return logits, labels, preds

    def _write_validation_artifacts(self, prefix, labels, preds):
        if self.save_dir is None:
            return

        all_targets = labels.numpy().tolist()
        all_preds = preds.numpy().tolist()

        report = classification_report(all_targets, all_preds, target_names=CLASS_NAMES, digits=3, zero_division=0)
        with open(os.path.join(self.save_dir, f"{prefix}_classification_report.txt"), "w") as f:
            f.write(report)

        cm = confusion_matrix(all_targets, all_preds)
        cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
        cm_path = os.path.join(self.save_dir, f"{prefix}_confusion_matrix.csv")
        cm_df.to_csv(cm_path)

        output_csv = os.path.join(self.save_dir, f"{prefix}_predictions.csv")
        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image_id", "true_label", "predicted_label"])
            if hasattr(self, "val_image_ids") and self.val_image_ids is not None:
                for i, (true, pred) in enumerate(zip(all_targets, all_preds)):
                    image_id = self.val_image_ids[i] if i < len(self.val_image_ids) else ""
                    writer.writerow([image_id, true, pred])
            elif hasattr(self, "val_indices") and hasattr(self, "dataset"):
                for i, (true, pred) in enumerate(zip(all_targets, all_preds)):
                    image_id = self.dataset[self.val_indices[i]]["image_id"]
                    writer.writerow([image_id, true, pred])
            else:
                for i, (true, pred) in enumerate(zip(all_targets, all_preds)):
                    writer.writerow([i, true, pred])

    def on_fit_end(self):
        if not self.val_outputs:
            return
        _logits, labels, preds = self._collect_validation_outputs()
        all_preds = preds.numpy().tolist()
        all_targets = labels.numpy().tolist()
        val_acc = accuracy_score(all_targets, all_preds)
        print(f"\nFinal Validation accuracy: {val_acc:.4f}")
        if self.save_dir is not None:
            print(f"Best validation overall accuracy achieved: {self.best_val_oa:.4f}")

        print("\n=== Per-Class Performance Metrics ===")
        report = classification_report(all_targets, all_preds, target_names=CLASS_NAMES, digits=3, zero_division=0)
        print(report)

        if self.save_dir is not None:
            self._write_validation_artifacts("last_val_epoch", labels, preds)
            print(f"Last validation epoch artifacts saved in {self.save_dir}")

            metrics_path = os.path.join(self.save_dir, "metrics.csv")
            try:
                from metrics import summarize_metrics
                summarize_metrics(metrics_path)
            except ImportError:
                print("Warning: summarize_metrics module not available")
        self.val_outputs.clear()

    def configure_optimizers(self):
        trainable_params = [param for param in self.parameters() if param.requires_grad]
        optimizer = optim.Lamb(trainable_params, lr=self.lr, weight_decay=1e-4)

        warmup_epochs = self.warmup_epochs
        if warmup_epochs == 0:
            scheduler = torch_optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.trainer.max_epochs, eta_min=1e-6
            )
        else:
            scheduler = torch_optim.lr_scheduler.SequentialLR(
                optimizer, 
                [
                    torch_optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs),
                    torch_optim.lr_scheduler.CosineAnnealingLR(
                        optimizer,
                        T_max=self.trainer.max_epochs - warmup_epochs,
                        eta_min=1e-6,
                    ),
                ],
                milestones=[warmup_epochs]
            )
            
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            }
        }


def get_model(num_classes, lr=1e-3, save_dir=None, freeze_backbone=False, warmup_epochs=0):
    model = LitClassifier(
        num_classes=num_classes,
        lr=lr,
        freeze_resnet=freeze_backbone,
        warmup_epochs=warmup_epochs,
    )
    model.save_dir = save_dir
    return model


