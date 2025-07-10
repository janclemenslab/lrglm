import torch
from torch import nn, optim
import lightning as L
from typing import Optional
from torch.nn.utils import parametrize


class NonNegative(nn.Module):
    def forward(self, X):
        X[X < 0.0] = 0.0
        return X


class Binary(nn.Module):
    def forward(self, X):
        return (X > 0).float()


class Unique(nn.Module):
    def forward(self, X):
        return (X == X.max()).float()


parameterizations = {
    "nonegative": NonNegative,
    "binary": Binary,
    "unique": Unique,
    "orthogonal": torch.nn.utils.parametrizations.orthogonal,
}


class LogisticRegressionModel(L.LightningModule):
    def __init__(
        self,
        input_dim,
        latent_dim,
        output_dim,
        task_ids,
        batch_size,
        lr=1e-3,
        l1_total=0.1,
        l1_encoder=1,
        l1_decoder=1,
        encoder_constraint: Optional[str] = None,
        decoder_constraint: Optional[str] = None,
        bias: bool = True,
    ):
        super().__init__()
        self.task_ids = task_ids
        self.batch_size = batch_size
        self.decoder_constraint = decoder_constraint
        self.encoder_constraint = encoder_constraint
        self.bias = bias

        self.example_data = (torch.rand((3, input_dim)), list(range(len(self.task_ids))))
        # linear projection onto shared low dimensional latent space
        # self.encoder = nn.Linear(input_dim, latent_dim, bias=True)
        # mult = 2
        self.encoder = nn.Sequential(
            # nn.Linear(input_dim, mult * latent_dim, bias=self.bias),
            # nn.Linear(mult * latent_dim, latent_dim, bias=self.bias),
            nn.Linear(input_dim, latent_dim, bias=self.bias),
        )

        if encoder_constraint is not None:
            parametrize.register_parametrization(
                self.encoder,
                "weight",
                parameterizations[self.encoder_constraint](self.encoder),
            )

        # per-species decoders from the latent space to the outputs
        self.decoder = []
        for task_num, task_id in enumerate(self.task_ids):
            self.decoder.append(nn.Linear(latent_dim, output_dim[task_id], bias=self.bias))
            if decoder_constraint is not None:
                parametrize.register_parametrization(
                    self.decoder[-1],
                    "weight",
                    parameterizations[self.decoder_constraint](),
                )

        self.decoder = nn.ModuleList(self.decoder)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.criterion = nn.NLLLoss()
        self.l1_total = l1_total
        self.l1_encoder = l1_encoder
        self.l1_decoder = l1_decoder
        self.lr = lr

        self.save_hyperparameters()

    # def L1_regularized_loss(self):
    #     reg_loss = 0
    #     for param in self.parameters():
    #         reg_loss += param.abs().mean()
    #     return reg_loss

    def L1_regularized_loss(self):
        reg_loss = 0
        for name, param in self.named_parameters():
            p = param.abs().mean()
            if 'encoder' in name:
                p *= self.l1_encoder
            if 'decoder' in name:
                p *= self.l1_decoder
            reg_loss += p
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
        x = self.decoder[task_id[0]](x)
        x = self.log_softmax(x)
        return x

    def step(self, batch):
        x, y, task_id = batch
        out = self.forward(x, task_id)
        loss = self.criterion(out, y) + self.l1_total * self.L1_regularized_loss()
        return out, loss

    def training_step(self, batch):
        out, loss = self.step(batch)
        self.log_dict({"train_loss": loss}, prog_bar=True, batch_size=self.batch_size)
        return loss

    def validation_step(self, batch):
        out, loss = self.step(batch)
        self.log_dict({"val_loss": loss}, prog_bar=True, batch_size=self.batch_size)
        return loss

    def test_step(self, batch):
        out, loss = self.step(batch)
        self.log_dict({"test_loss": loss}, batch_size=self.batch_size)
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
