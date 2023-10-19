import wandb
from wandb.sdk.wandb_run import Run

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
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
    encoder_lr_decay_rate: float = 0.1
    decoder_lr_decay_rate: float = 0.1
    encoder_lr_decay_step: int = 30
    decoder_lr_decay_step: int = 30
    
    def __new__(cls, *args, **kwargs):
        try:
            initializer = cls.__initializer
        except AttributeError:
            # Store the original init on the class in a different place
            cls.__initializer = initializer = cls.__init__
            # replace init with something harmless
            cls.__init__ = lambda *a, **k: None

        # code from adapted from Arne
        added_args = {}
        for name in list(kwargs.keys()):
            if name not in cls.__annotations__:
                added_args[name] = kwargs.pop(name)

        ret = object.__new__(cls)
        initializer(ret, **kwargs)
        # ... and add the new ones by hand
        for new_name, new_val in added_args.items():
            setattr(ret, new_name, new_val)

        return ret


class VIDNetPLWrapper(LightningModule):
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

        # self.mask_soft_iou_loss = SoftIoULoss()

        self.train_f1 = BinaryF1Score()
        self.train_jaccard = BinaryJaccardIndex()
        self.val_f1 = BinaryF1Score()
        self.val_jaccard = BinaryJaccardIndex()

        self.save_hyperparameters()

        self.automatic_optimization = False

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
    
    @torch.no_grad()
    def log_loc_output(self, x, x_ela, gt_mask, pred_mask, step_idx):
        x = x.detach().cpu()
        x_ela = x_ela.detach().cpu()
        gt_mask = gt_mask.float().detach().cpu()
        pred_mask = pred_mask.float().detach().cpu()
        logger = self.logger.experiment
        if isinstance(logger, Run):
            log_images = []
            log_images.append(wandb.Image(x, caption="input"))
            log_images.append(wandb.Image(x_ela, caption="ela"))
            log_images.append(wandb.Image(gt_mask, caption="gt_mask"))
            log_images.append(wandb.Image(pred_mask, caption="pred_mask"))
            logger.log({"train_loc_output": log_images}, step=step_idx)
        elif isinstance(logger, SummaryWriter):
            logger.add_images("train_loc_x", x, dataformats="CHW", global_step=step_idx)
            logger.add_images("train_loc_x_ela", x_ela, dataformats="CHW", global_step=step_idx)
            logger.add_images("train_loc_gt", gt_mask, dataformats="HW", global_step=step_idx)
            logger.add_images("train_loc_pred", pred_mask, dataformats="HW", global_step=step_idx)
        else:
            pass

    def training_step(self, batch, batch_idx):
        enc_opt, dec_opt = self.optimizers()
        x, x_ela, gt_mask, label = batch
        B, T, C, H, W = x.shape

        loss = 0
        for b in range(B):
            y_hat = self(x[b], x_ela[b])
            # loss += self.mask_soft_iou_loss(y_hat, gt_mask[b])
            loss += F.binary_cross_entropy(y_hat, gt_mask[b])

            self.train_f1.update(y_hat, gt_mask[b])
            self.train_jaccard.update(y_hat, gt_mask[b])

        if self.global_step % 200 == 0:
            self.log_loc_output(x[b][T-1], x_ela[b][T-1], gt_mask[b][T-1], y_hat[T-1], self.global_step)

        enc_opt.zero_grad()
        dec_opt.zero_grad()
        self.manual_backward(loss)
        enc_opt.step()
        dec_opt.step()
        # if batch_idx % 2 == 0:
        #     enc_opt.zero_grad()
        # else:
        #     dec_opt.zero_grad()
        # self.manual_backward(loss)
        
        # if batch_idx % 2 == 0:
        #     enc_opt.step()
        # else:
        #     dec_opt.step()

        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        self.log("train_f1", self.train_f1, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(
            "train_jaccard", self.train_jaccard, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True
        )

    def validation_step(self, batch, batch_idx):
        x, x_ela, gt_mask, label = batch
        B, T, C, H, W = x.shape
        
        loss = 0
        for b in range(B):
            y_hat = self(x[b], x_ela[b])
            # loss += self.mask_soft_iou_loss(y_hat, gt_mask[b])
            loss += F.binary_cross_entropy(y_hat, gt_mask[b])

            self.val_f1.update(y_hat, gt_mask[b])
            self.val_jaccard.update(y_hat, gt_mask[b])

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

        encoder_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer, step_size=self.training_config.encoder_lr_decay_step, gamma=self.training_config.encoder_lr_decay_rate)
        decoder_scheduler = torch.optim.lr_scheduler.StepLR(decoder_optimizer, step_size=self.training_config.decoder_lr_decay_step, gamma=self.training_config.decoder_lr_decay_rate)

        return [encoder_optimizer, decoder_optimizer], [encoder_scheduler, decoder_scheduler]
