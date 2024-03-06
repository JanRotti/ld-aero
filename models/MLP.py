import lightning as pl
import torch
import torch.nn as nn

class MLP(pl.LightningModule):
  
  def __init__(self,
      input_dim=(1, 28, 28),
      layer_dims=[128, 64, 32],
      output_dim=10,
      final_activation=nn.Identity(),
      activation=nn.SiLU(),
      image_key="image",
      target_key="target",
      learning_rate=1e-4,
      loss=nn.MSELoss(),
      ):
    super().__init__()
    assert(len(layer_dims) > 0,"'layer_dims' must have at least one element.")
    self.flatten = nn.Flatten()
    self.input_dim = input_dim
    self.activation = activation
    self.final_activation = final_activation
    self.image_key = image_key
    self.target_key = target_key
    self.learning_rate = learning_rate
    self.loss = loss

    self.num_features = torch.prod(torch.tensor(input_dim)).item()
    curr_dim = self.num_features
    
    layers = []
    for dim in layer_dims:
      layers.append(nn.Linear(curr_dim, dim))
      layers.append(self.activation)
      curr_dim = dim

    self.mid = nn.Sequential(*layers)
    self.final = nn.Linear(curr_dim, output_dim)
    self.final_activation = final_activation
    
  def forward(self, x):
    x = self.flatten(x)
    x = self.mid(x)
    x = self.final(x)
    return self.final_activation(x)
  
  def training_step(self, batch, batch_idx):
    x = self.get_input(batch, self.image_key)
    b = x.shape[0]
    y = self.get_input(batch, self.target_key)
    y_hat = self(x)
    loss = self.loss(y_hat, y)
    self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=b)
    return loss
  
  def validation_step(self, batch, batch_idx):
    x = self.get_input(batch, self.image_key)
    b = x.shape[0]
    y = self.get_input(batch, self.target_key)
    y_hat = self(x)
    loss = self.loss(y_hat, y)
    self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=b)
    return loss
  
  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    return optimizer
  
  def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        return x