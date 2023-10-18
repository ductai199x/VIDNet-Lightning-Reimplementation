import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex

from modules.model import RSIS, FeatureExtractor
from utils.objectives import SoftIoULoss

from dataclasses import dataclass
from typing import *


@dataclass
class EncoderConfig:
    base_model: Literal["resnet101", "resnet50", "resnet34", "vgg16", "vgg16_bn", "unet"] = "vgg16_bn"
    dropout: float = 0.5
    hidden_size: int = 512
    kernel_size: int = 3  # Convolution kernel size
    input_dim: int = 3  # Input image channel
    use_gpu: bool = False


@dataclass
class DecoderConfig:
    dropout: float = 0.5
    hidden_size: int = 512
    kernel_size: int = 3  # Convolution kernel size
    skip_mode: Literal["sum", "concat", "mul", "none"] = "concat"
    maxseqlen: int = 5  # Maximum sequence length
    use_gpu: bool = False


@dataclass
class TrainingConfig:
    encoder_lr: float = 1.0e-4
    encoder_weight_decay: float = 5.0e-5
    decoder_lr: float = 1.0e-3
    decoder_weight_decay: float = 5.0e-5


class VIDNet(LightningModule):
    def __init__(
        self,
        encoder_config: EncoderConfig,
        decoder_config: DecoderConfig,
        training_config: TrainingConfig,
    ):
        super().__init__()
        self.encoder = FeatureExtractor(encoder_config)
        self.decoder = RSIS(decoder_config)
        self.training_config = training_config

        self.mask_soft_iou_loss = SoftIoULoss()

        self.train_f1 = BinaryF1Score()
        self.train_jaccard = BinaryJaccardIndex()
        self.val_f1 = BinaryF1Score()
        self.val_jaccard = BinaryJaccardIndex()

    def forward(self, x, x_ela):
        T, C, H, W = x.shape
        features = self.encoder(x, x_ela=x_ela)

        output_masks = []
        spatial_hidden_states = None
        temporal_hidden_states = None
        for t in range(T):
            _, output_mask_t, hidden_states = self.decoder(
                features, spatial_hidden_states, temporal_hidden_states, T=t
            )
            spatial_hidden_states = hidden_states
            temporal_hidden_states = []
            for state in hidden_states:
                temporal_hidden_states.append(state[0])

            output_mask_t = F.interpolate(output_mask_t, size=(H, W), mode="bilinear")  # 1, 1, H, W
            output_masks.append(output_mask_t)

        output_masks = torch.cat(output_masks, dim=1)  # 1, T, H, W
        output_mask = self.decoder.conv_out(output_masks).squeeze()  # 1, 1, H, W -> H, W
        output_mask = torch.sigmoid(output_mask)
        return output_mask

    def training_step(self, batch, batch_idx):
        x, x_ela, y = batch
        y_hat = self(x, x_ela)
        loss = self.mask_soft_iou_loss(y_hat, y)

        print(loss.shape, loss)

        self.train_f1.update(y_hat, y)
        self.train_jaccard.update(y_hat, y)

        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        self.log("train_f1", self.train_f1, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(
            "train_jaccard", self.train_jaccard, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x, x_ela, y = batch
        y_hat = self(x, x_ela)
        loss = self.mask_soft_iou_loss(y_hat, y)

        self.val_f1.update(y_hat, y)
        self.val_jaccard.update(y_hat, y)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(
            "val_jaccard", self.val_jaccard, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True
        )

    def configure_optimizers(self):
        encoder_optimizer = torch.optim.Adam(
            self.encoder.parameters(),
            lr=self.training_config.encoder_lr,
            weight_decay=self.training_config.encoder_weight_decay,
        )
        decoder_optimizer = torch.optim.Adam(
            self.decoder.parameters(),
            lr=self.training_config.decoder_lr,
            weight_decay=self.training_config.decoder_weight_decay,
        )

        encoder_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer, step_size=30, gamma=0.1)
        decoder_scheduler = torch.optim.lr_scheduler.StepLR(decoder_optimizer, step_size=30, gamma=0.1)

        return [encoder_optimizer, decoder_optimizer], [encoder_scheduler, decoder_scheduler]
