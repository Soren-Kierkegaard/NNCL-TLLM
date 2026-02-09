import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from collections import deque

from typing import Tuple, Optional, List

from .nncltllm import NNCLTLLM

def train_epoch(
	model: NNCLTLLM,
	dataloader: DataLoader,
	optimizer: torch.optim.Optimizer,
	device: torch.device,
	lambda_weight: float = 1.0
) -> dict:

	"""
	  Entraîne le modèle pour une époque
	"""

	# Set mode training
	model.train()

	# Init Loss
	total_loss = 0.0
	total_forecast_loss = 0.0
	total_proto_loss = 0.0
	total_nncl_loss = 0.0
	num_batches = 0

	for batch_x, batch_y in dataloader:
		batch_x = batch_x.to(device)
		batch_y = batch_y.to(device)

		optimizer.zero_grad()

		# Forward pass
		forecast, losses = model(batch_x)

		# Calcul des losses
		loss_forecast = F.mse_loss(forecast, batch_y.unsqueeze(2)) # MSE entre y et y_hat, batch_y(B, H) --> batch_y(B, H, 1) d'où unsqueeze sur pos 2
		loss_proto = losses['loss_proto']
		loss_nncl = losses['loss_nncl']

		# Loss totale : Ltotal = Lforecast + λ(LNNCL + Lproto) ((Eq. 8 de l'article))
		loss_total = loss_forecast + lambda_weight * (loss_nncl + loss_proto)

		# Backward pass
		loss_total.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
		optimizer.step()

		# Accumuler les losses
		total_loss			+= loss_total.item()
		total_forecast_loss += loss_forecast.item()
		total_proto_loss	+= loss_proto.item()
		total_nncl_loss		+= loss_nncl.item()
		num_batches			+= 1

	return {
		'total_loss': total_loss / num_batches,
		'forecast_loss': total_forecast_loss / num_batches,
		'proto_loss': total_proto_loss / num_batches,
		'nncl_loss': total_nncl_loss / num_batches
	}


def evaluate(model: NNCLTLLM, dataloader: DataLoader, device: torch.device) -> dict:

	"""
		Évalue le modèle
	"""
	
	model.eval()

	total_mse = 0.0
	total_mae = 0.0
	num_batches = 0

	predictions = []
	ground_truths = []
	inputs = []

	with torch.no_grad():
		for batch_x, batch_y in dataloader:
			batch_x = batch_x.to(device)
			batch_y = batch_y.to(device)

			forecast, _ = model(batch_x)

			mse = F.mse_loss(forecast, batch_y)
			mae = F.l1_loss(forecast, batch_y)

			total_mse += mse.item()
			total_mae += mae.item()
			num_batches += 1

			# Sauvegarder pour visualisation
			predictions.append(forecast.cpu())
			ground_truths.append(batch_y.cpu())
			inputs.append(batch_x.cpu())

	return {
		'mse': total_mse / num_batches,
		'mae': total_mae / num_batches,
		'predictions': torch.cat(predictions, dim=0),
		'ground_truths': torch.cat(ground_truths, dim=0),
		'inputs': torch.cat(inputs, dim=0)
	}

#