import lightning as L
import torch
from torch import nn, optim
from typing import Dict, Any
from torchmetrics.classification import MulticlassAccuracy


class LN_LR_Encoder(nn.Module):
    """Multiple filter banks, project onto low dimensional subspace"""

    def __init__(
        self, nb_feat: int, latent_dim: int, block_len: int, bias: bool = False
    ):
        super().__init__()
        self.nb_feat = nb_feat
        self.latent_dim = latent_dim
        self.block_len = block_len
        self.bias = bias

        self.filters = nn.Conv1d(
            self.nb_feat,
            self.latent_dim * self.nb_feat,
            kernel_size=self.block_len,
            groups=self.nb_feat,
            bias=self.bias,
        )
        self.project = nn.Conv1d(
            self.latent_dim,
            self.latent_dim,
            kernel_size=self.nb_feat,
            groups=self.latent_dim,
            bias=self.bias,
        )

        self.example_data = torch.randn(7, self.nb_feat, self.block_len)
        self.output_shape = (
            self.latent_dim
        )  # [None, *self.forward(self.example_data).shape[1:]]

    def forward(self, x):
        x = self.filters(x)
        x = torch.tanh(x)
        x = x.reshape((-1, self.nb_feat, self.latent_dim))
        x = self.project(x.transpose(2, 1))[..., 0]
        return x


class LN_what_Encoder(nn.Module):
    """Multiple filter banks, project onto low dimensional subspace, V2"""

    def __init__(
        self, nb_feat: int, latent_dim: int, block_len: int, bias: bool = False
    ):
        super().__init__()
        self.nb_feat = nb_feat
        self.latent_dim = latent_dim
        self.block_len = block_len
        self.bias = bias

        self.encoder = nn.Sequential(
            nn.Conv1d(
                self.nb_feat,
                self.latent_dim * self.nb_feat,
                kernel_size=self.block_len,
                groups=self.nb_feat,
                bias=self.bias,
            ),
            nn.Tanh(),
            nn.Flatten(1),
            nn.Unflatten(-1, (self.nb_feat, self.latent_dim)),
            nn.Conv1d(self.nb_feat, 1, 1),
            nn.Flatten(1),
        )

        self.example_data = torch.randn(7, self.nb_feat, self.block_len)
        self.output_shape = (
            self.latent_dim
        )  # [None, *self.forward(self.example_data).shape[1:]]

    def forward(self, x):
        return self.encoder(x)


class LN_full_Encoder(nn.Module):
    """Multiple filter banks, full space, V2"""

    def __init__(
        self, nb_feat: int, latent_dim: int, block_len: int, bias: bool = False
    ):
        super().__init__()
        self.nb_feat = nb_feat
        self.latent_dim = latent_dim
        self.block_len = block_len
        self.bias = bias

        self.encoder = nn.Sequential(
            nn.Conv1d(
                self.nb_feat,
                self.latent_dim * self.nb_feat,
                kernel_size=self.block_len,
                groups=self.nb_feat,
                bias=self.bias,
            ),
            nn.Tanh(),
            nn.Flatten(1),
        )
        self.example_data = torch.randn(7, self.nb_feat, self.block_len)
        self.output_shape = self.nb_feat * self.latent_dim

    def forward(self, x):
        return self.encoder(x)


class LN_Simple_Encoder(nn.Module):
    """Linear"""

    def __init__(
        self, nb_feat: int, latent_dim: int, block_len: int, bias: bool = False
    ):
        super().__init__()
        self.nb_feat = nb_feat
        self.latent_dim = latent_dim
        self.block_len = block_len
        self.bias = bias

        self.encoder = nn.Sequential(
            # nn.Conv1d(self.nb_feat, self.latent_dim * self.nb_feat, kernel_size=self.block_len, groups=self.nb_feat, bias=self.bias),
            # nn.Tanh(),
            nn.Flatten(1),
            nn.Linear(self.nb_feat * self.block_len, self.latent_dim, bias=self.bias),
            nn.Flatten(1),
        )
        self.example_data = torch.randn(7, self.nb_feat, self.block_len)
        self.output_shape = self.latent_dim

    def forward(self, x):
        return self.encoder(x)


class LogisticRegressionModel(L.LightningModule):
    def __init__(
        self,
        latent_dim: int,
        output_dims: Dict[Any, int],
        batch_size: int,
        encoder: nn.Module,
        lr: float = 1e-3,
        l1_weight: float = 0.1,
        bias: bool = False,
        block_len: int = 6,
        nb_feat: int = 19,
    ):
        super().__init__()
        self.task_ids = list(output_dims.keys())
        self.batch_size = batch_size
        self.bias = bias
        self.block_len = block_len
        self.nb_feat = nb_feat
        self.latent_dim = latent_dim

        self.encoder = encoder

        # per-species decoders from the latent space to the outputs
        self.decoder = []
        for task_id in self.task_ids:
            self.decoder.append(
                nn.Linear(
                    self.encoder.output_shape, output_dims[task_id], bias=self.bias
                )
            )

        self.decoder = nn.ModuleList(self.decoder)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.criterion = nn.NLLLoss()
        self.acc = MulticlassAccuracy(num_classes=3, top_k=1, average='macro', multidim_average='global')
        self.l1_weight = l1_weight
        self.lr = lr

        self.save_hyperparameters()  # (ignore=['encoder'])

        self.example_data = (torch.randn((3, self.nb_feat, self.block_len)), [0])
        # self.output_shape = [None, *self.forward(self.example_data, task_id=[0]).shape[1:]]

    def L1_regularized_loss(self):
        reg_loss = 0
        for param in self.parameters():
            reg_loss += param.abs().mean()
        return reg_loss

    def get_weights(self, apply_constraints=True):
        # get the original weights, pre constraint
        encoder_params = {
            name: param.detach() for name, param in self.encoder.named_parameters()
        }
        decoder_params = {
            name: param.detach() for name, param in self.decoder.named_parameters()
        }

        # to numpy
        for name in list(encoder_params.keys()):
            new_name = name.replace("parametrizations.", "").replace(".original", "")
            w = encoder_params[name]
            try:
                w = w.numpy()
            except:
                pass
            encoder_params[new_name] = w

        for name in list(decoder_params.keys()):
            new_name = name.replace("parametrizations.", "").replace(".original", "")
            w = decoder_params[name]
            try:
                w = w.numpy()
            except:
                pass
            decoder_params[new_name] = w

        weights = {"encoder": encoder_params, "decoder": decoder_params}
        return weights

    def forward(self, x, task_id):
        x = self.encoder(x)
        # x = x[..., 0]
        # x = self.nls(x)
        x = self.decoder[task_id[0]](x)
        x = self.log_softmax(x)
        return x

    def step(self, batch):
        x, y, task_id = batch
        out = self.forward(x, task_id)
        loss = self.criterion(out, y) + self.l1_weight * self.L1_regularized_loss()
        acc = self.acc(out, y)
        return out, loss, acc

    def training_step(self, batch):
        out, loss, acc = self.step(batch)
        self.log_dict({"train_loss": loss, "train_acc": acc}, prog_bar=True, batch_size=self.batch_size)
        return loss

    def validation_step(self, batch):
        out, loss, acc = self.step(batch)
        self.log_dict({"val_loss": loss, "val_acc": acc}, prog_bar=True, batch_size=self.batch_size)
        return loss

    def test_step(self, batch):
        out, loss, acc = self.step(batch)
        self.log_dict({"test_loss": loss, "test_acc": acc}, batch_size=self.batch_size)
        return loss

    def predict_step(self, batch):
        # out, loss = self.step(batch)
        x, y, task_id = batch
        out = self.forward(x, task_id)
        return out

    def configure_optimizers(self):
        # optimizer = optim.Adam(self.parameters(), lr=1e-3)#, weight_decay=1e-5)
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        # optimizer = optim.LBFGS(self.parameters(), lr=self.lr, max_iter=1_000)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=5,
            cooldown=2,
            min_lr=1e-8,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
