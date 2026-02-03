import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import itertools
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
#import seaborn as sns
from pathlib import Path

""" Note perso:
@dataclass : This module provides a decorator and functions for automatically adding generated special methods such as
	__init__() and __repr__() to user-defined classes. It was originally described in PEP 557.

	@dataclass
	class InventoryItem:
		#Class for keeping track of an item in inventory.
		name: str
		unit_price: float
		quantity_on_hand: int = 0

		def total_cost(self) -> float:
			return self.unit_price * self.quantity_on_hand

	The __init__ method will be auto added to the class
"""
@dataclass(init=True, repr=True, eq=True, order=False, frozen=False) # params par défaut
class HyperparameterConfig:

	"""
		Configuration des hyperparamètres à optimiser
	"""

	# Architecture
	patch_len: int = 5
	stride: int = 2
	num_prototypes: int = 1000
	support_set_size: int = 10000
	top_k: int = 8
	embedding_dim: int = 768
	num_llm_layers: int = 6

	# Entraînement
	learning_rate: float = 1e-3
	batch_size: int = 16
	lambda_weight: float = 1.0
	weight_decay: float = 1e-4

	# Données (fixés généralement)
	input_length: int = 10
	pred_length: int = 1
	num_channels: int = 1

	"""
		a méthode __post_init__() est appelée après l’exécution de __init__() générée automatiquement par @dataclass.
		Elle permet de réaliser des traitements ou des vérifications personnalisées sur les attributs.
	"""
	def __post_init__(self):
		"""Validation des hyperparamètres"""
		if self.patch_len >= self.input_length:
			print(f"❌ ValueError - patch_len ({self.patch_len}) doit être < input_length ({self.input_length})")

			if (self.input_length - 1) <= 0:
			  raise ValueError(f"patch_len ({self.patch_len}) doit être < input_length ({self.input_length})")
			else:
			  self.patch_len = int(self.input_length - 1)
			  print(f"Prendre une valeur inférieur ({self.input_length}) --> ({self.patch_len})")


		num_patches = (self.input_length - self.patch_len) // self.stride + 1
		if num_patches < 1:
			raise ValueError(f"Configuration invalide: produit {num_patches} patches")
			#print(f"Configuration invalide: produit {num_patches} patches")

	def to_dict(self):
		return asdict(self)

	def show(self):
		print(f"patch_len - {type(self.patch_len)}")
		print(f"stride - {type(self.stride)}")
		print(f"num_prototypes - {type(self.num_prototypes)}")
		print(f"support_set_size - {type(self.support_set_size)}")
		print(f"top_k - {type(self.top_k)}")
		print(f"batch_size - {type(self.batch_size)}")


@dataclass
class TrialResult:
	"""Résultat d'un essai d'hyperparamètres"""
	config: HyperparameterConfig
	train_mse: float
	val_mse: float
	val_mae: float
	train_time: float
	num_params: int
	trial_id: int
	timestamp: str

	def score(self, metric: str = 'val_mse') -> float:
		"""Score à minimiser"""
		if metric == 'val_mse':
			return self.val_mse
		elif metric == 'val_mae':
			return self.val_mae
		elif metric == 'combined':
			return 0.5 * self.val_mse + 0.5 * self.val_mae
		else:
			raise ValueError(f"Métrique inconnue: {metric}")


class HyperparameterTuner:

	"""
		Classe principale pour l'optimisation des hyperparamètres

		Choix de design:
		1. Modularité: Séparation entre recherche, entraînement et évaluation
		2. Reproductibilité: Seed fixé, sauvegarde complète des configs
		3. Efficacité: Early stopping, validation rapide
		4. Traçabilité: Logs détaillés, sauvegarde de tous les résultats
	"""

	def __init__(
		self,
		data,
		train_dataset,
		val_dataset,
		channel_names: List[str],
		device: str = 'cuda',
		results_dir: str = './hyperparameter_search',
		seed: int = 42
	):
		self.data = data
		self.train_dataset = train_dataset
		self.val_dataset = val_dataset
		self.channel_names = channel_names
		self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
		self.results_dir = Path(results_dir)
		self.results_dir.mkdir(parents=True, exist_ok=True)
		self.seed = seed

		# Fixer la seed pour reproductibilité
		torch.manual_seed(seed)
		np.random.seed(seed)
		if torch.cuda.is_available():
			torch.cuda.manual_seed(seed)

		self.results: List[TrialResult] = []
		self.best_result: Optional[TrialResult] = None

	def train_and_evaluate(
		self,
		config: HyperparameterConfig,
		trial_id: int,
		max_epochs: int = 20,
		patience: int = 5,
		verbose: bool = True
	) -> TrialResult:
		"""
		Entraîne et évalue un modèle avec une configuration donnée

		Choix:
		- max_epochs réduit (20) pour rapidité
		- Early stopping pour éviter l'overfitting et gagner du temps
		- Validation à chaque époque pour détecter rapidement les mauvaises configs
		"""
		if verbose:
			print(f"\n{'='*70}")
			print(f"TRIAL {trial_id}")
			print(f"{'='*70}")
			print("Configuration:")
			for key, value in config.to_dict().items():
				print(f"  {key}: {value}")

		start_time = datetime.now()

		# Créer les dataloaders
		train_loader = DataLoader(
			self.train_dataset,
			batch_size= int(config.batch_size), # np.random.choice() retourne un type numpy (comme numpy.int64) au lieu d'un int --> force le cast
			shuffle=True,
			num_workers=0  # 0 pour éviter des problèmes de multiprocessing
		)

		val_loader = DataLoader(
			self.val_dataset,
			batch_size = int(config.batch_size), # np.random.choice() retourne un type numpy (comme numpy.int64) au lieu d'un int
			shuffle=False,
			num_workers=0
		)

		#print(f"Type Params")
		#config.show()

		# Créer le modèle
		try:
			#from NNCL_TLLM import NNCLTLLM	 # Ajuster l'import selon votre structure

			model = NNCLTLLM(
				num_channels = config.num_channels,
				input_length = config.input_length,
				pred_length	 = config.pred_length,
				patch_len	 = config.patch_len,
				stride		 = config.stride,
				num_prototypes = config.num_prototypes,
				support_set_size = config.support_set_size,
				top_k = config.top_k,
				embedding_dim = config.embedding_dim,
				num_llm_layers = config.num_llm_layers,
				freeze_llm = True
			).to(self.device)
		except Exception as e:
			print(f"❌ Erreur lors de la création du modèle: {e}")
			# Retourner un résultat avec des scores très mauvais
			return TrialResult(
				config=config,
				train_mse=float('inf'),
				val_mse=float('inf'),
				val_mae=float('inf'),
				train_time=0,
				num_params=0,
				trial_id=trial_id,
				timestamp=datetime.now().isoformat()
			)

		# Compter les paramètres
		num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
		if verbose:
			print(f"\nParamètres entraînables: {num_params:,}")

		# Optimizer
		optimizer = torch.optim.AdamW(
			filter(lambda p: p.requires_grad, model.parameters()),
			lr=config.learning_rate,
			weight_decay=config.weight_decay
		)

		# Scheduler (optionnel mais recommandé)
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
			optimizer,
			mode='min',
			factor=0.5,
			patience=3,
		)

		# Early stopping
		best_val_loss = float('inf')
		patience_counter = 0
		best_model_state = None

		# Entraînement
		for epoch in range(max_epochs):
			# Train
			model.train()
			train_loss = 0.0
			num_batches = 0

			for batch_x, batch_y in train_loader:
				batch_x = batch_x.to(self.device)
				batch_y = batch_y.to(self.device)

				optimizer.zero_grad()

				try:
					forecast, losses = model(batch_x)
					#loss_forecast = torch.nn.functional.mse_loss(forecast, batch_y.unsqueeze(2)) # MSE entre y et y_hat, batch_y(B, H) --> batch_y(B, H, 1) d'où unsqueeze sur pos 2
					loss_forecast = F.mse_loss(forecast, batch_y.unsqueeze(2)) # MSE entre y et y_hat, batch_y(B, H) --> batch_y(B, H, 1) d'où unsqueeze sur pos 2
					loss_total = loss_forecast + config.lambda_weight * (
						losses['loss_nncl'] + losses['loss_proto']
					)

					loss_total.backward()
					torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
					optimizer.step()

					train_loss += loss_forecast.item()
					num_batches += 1

				except RuntimeError as e:
					print(f"❌ Erreur pendant l'entraînement: {e}")
					return TrialResult(
						config=config,
						train_mse=float('inf'),
						val_mse=float('inf'),
						val_mae=float('inf'),
						train_time=(datetime.now() - start_time).total_seconds(),
						num_params=num_params,
						trial_id=trial_id,
						timestamp=datetime.now().isoformat()
					)

			train_mse = train_loss / num_batches

			#print("== partie evaluation")
			# Validation
			model.eval()
			val_mse_total = 0.0
			val_mae_total = 0.0
			val_batches = 0

			with torch.no_grad():
				for batch_x, batch_y in val_loader:
					batch_x = batch_x.to(self.device)
					batch_y = batch_y.to(self.device)

					forecast, _ = model(batch_x)

					val_mse_total += F.mse_loss(forecast, batch_y.unsqueeze(2)).item()
					val_mae_total += torch.nn.functional.l1_loss(forecast, batch_y.unsqueeze(2)).item()
					val_batches += 1

			val_mse = val_mse_total / val_batches
			val_mae = val_mae_total / val_batches

			# Scheduler step
			scheduler.step(val_mse)

			if verbose and (epoch % 5 == 0 or epoch == max_epochs - 1):
				print(f"Epoch {epoch+1}/{max_epochs} - "
					  f"Train MSE: {train_mse:.6f}, "
					  f"Val MSE: {val_mse:.6f}, "
					  f"Val MAE: {val_mae:.6f}")

			# Early stopping
			if val_mse < best_val_loss:
				best_val_loss = val_mse
				best_model_state = model.state_dict().copy()
				patience_counter = 0
			else:
				patience_counter += 1
				if patience_counter >= patience:
					if verbose:
						print(f"Early stopping à l'époque {epoch+1}")
					break

		# Temps d'entraînement
		train_time = (datetime.now() - start_time).total_seconds()

		# Évaluation finale avec le meilleur modèle
		if best_model_state is not None:
			model.load_state_dict(best_model_state)

		model.eval()
		final_val_mse = 0.0
		final_val_mae = 0.0
		val_batches = 0

		with torch.no_grad():
			for batch_x, batch_y in val_loader:
				batch_x = batch_x.to(self.device)
				batch_y = batch_y.to(self.device)

				forecast, _ = model(batch_x)

				final_val_mse += torch.nn.functional.mse_loss(forecast, batch_y.unsqueeze(2)).item()
				final_val_mae += torch.nn.functional.l1_loss(forecast, batch_y.unsqueeze(2)).item()
				val_batches += 1

		final_val_mse /= val_batches
		final_val_mae /= val_batches

		result = TrialResult(
			config=config,
			train_mse=train_mse,
			val_mse=final_val_mse,
			val_mae=final_val_mae,
			train_time=train_time,
			num_params=num_params,
			trial_id=trial_id,
			timestamp=datetime.now().isoformat()
		)

		if verbose:
			print(f"\n✓ Trial {trial_id} terminé en {train_time:.1f}s")
			print(f"  Final Val MSE: {final_val_mse:.6f}")
			print(f"  Final Val MAE: {final_val_mae:.6f}")

		return result

	def grid_search(
		self,
		param_grid: Dict[str, List],
		max_epochs: int = 20,
		patience: int = 5,
		max_trials: Optional[int] = None
	) -> List[TrialResult]:
		"""
		Grid Search exhaustif

		Choix:
		- Exploration complète de toutes les combinaisons
		- Bon pour espaces de recherche petits/moyens
		- Garantit de trouver le meilleur dans la grille

		Args:
			param_grid: Dict avec listes de valeurs à tester
			max_epochs: Nombre max d'époques par trial
			patience: Patience pour early stopping
			max_trials: Limite le nombre de trials (None = tous)

		Returns:
			Liste de tous les résultats triés par performance
		"""
		print(f"\n{'='*70}")
		print("GRID SEARCH")
		print(f"{'='*70}")

		# Générer toutes les combinaisons
		keys = list(param_grid.keys())
		values = list(param_grid.values())
		combinations = list(itertools.product(*values))

		total_trials = len(combinations)
		if max_trials:
			total_trials = min(total_trials, max_trials)
			combinations = combinations[:max_trials]

		print(f"Nombre total de combinaisons: {total_trials}")
		print(f"Espace de recherche:")
		for key, vals in param_grid.items():
			print(f"  {key}: {vals}")

		# Tester toutes les combinaisons
		for trial_id, combo in enumerate(combinations, 1):
			# Créer la config
			config_dict = dict(zip(keys, combo))

			# Compléter avec les valeurs par défaut
			base_config = HyperparameterConfig(
				input_length=self.train_dataset.input_length,
				pred_length=self.train_dataset.pred_length,
				num_channels=self.train_dataset.num_channels
			)

			# Mettre à jour avec les paramètres de la grille
			for key, value in config_dict.items():
				setattr(base_config, key, value)

			# Entraîner et évaluer
			result = self.train_and_evaluate(
				config=base_config,
				trial_id=trial_id,
				max_epochs=max_epochs,
				patience=patience,
				verbose=True
			)

			self.results.append(result)

			# Mettre à jour le meilleur résultat
			if self.best_result is None or result.val_mse < self.best_result.val_mse:
				self.best_result = result
				print(f"\n🏆 Nouveau meilleur résultat! Val MSE: {result.val_mse:.6f}")

			# Sauvegarder les résultats intermédiaires
			self.save_results()

		# Trier les résultats
		self.results.sort(key=lambda x: x.val_mse)

		return self.results

	def random_search(
		self,
		param_distributions: Dict[str, Tuple],
		n_trials: int = 50,
		max_epochs: int = 20,
		patience: int = 5
	) -> List[TrialResult]:
		"""
		Random Search

		Choix:
		- Plus efficace que Grid Search pour grands espaces
		- Explore mieux les espaces continus (learning rate, etc.)
		- Recommandé comme première approche

		Args:
			param_distributions: Dict avec (min, max) ou liste de valeurs
			n_trials: Nombre d'essais aléatoires
			max_epochs: Nombre max d'époques par trial
			patience: Patience pour early stopping

		Returns:
			Liste de résultats triés
		"""
		print(f"\n{'='*70}")
		print("RANDOM SEARCH")
		print(f"{'='*70}")
		print(f"Nombre de trials: {n_trials}")
		print(f"Distributions:")
		for key, dist in param_distributions.items():
			print(f"  {key}: {dist}")

		for trial_id in range(1, n_trials + 1):
			# Échantillonner une configuration aléatoire
			config_dict = {}
			for key, dist in param_distributions.items():
				#if isinstance(dist, (list, tuple)) and len(dist) == 2:
				if isinstance(dist, tuple) and len(dist) == 2:
					# Range (min, max)
					if isinstance(dist[0], int):
						# Entier
						config_dict[key] = int(np.random.randint(dist[0], dist[1] + 1))
					else:
						# Float (échelle log pour learning rate)
						if key == 'learning_rate':
							config_dict[key] = 10 ** np.random.uniform(
								np.log10(dist[0]), np.log10(dist[1])
							)
						else:
							config_dict[key] = np.random.uniform(dist[0], dist[1])
				else:
					# Liste de choix
					config_dict[key] = np.random.choice(dist)

			# Créer le dataset, input_len ne peut pas être supérieur à la taille restante de validation
			input_length = config_dict['input_length']
			if input_length > len(self.data.loc[int(len(df)*.70):len(df), self.channel_names]):
			  #raise Exception(f"input_len : {input_length} et supérieur à la taille du validation_set de taille {len(self.data.loc[int(len(df)*.70):len(df), self.channel_names])}")
			  print(f"input_len : {input_length} et supérieur à la taille du validation_set de taille {len(self.data.loc[int(len(df)*.70):len(df), self.channel_names])}")
			  continue # Passer à la prochaine itération

			self.train_dataset = TimeSeriesDataset(self.data.loc[:int(len(df)*.70), self.channel_names].values, input_length, 1) # prendre 80%
			self.val_dataset   = TimeSeriesDataset(self.data.loc[int(len(df)*.70):len(df), self.channel_names].values, input_length, 1) # prendre 20% - Ne pas retirer input_len risque de set à 0 série pred

			# Créer la config
			try:
			  base_config = HyperparameterConfig(
				  input_length = self.train_dataset.input_length,
				  pred_length = self.train_dataset.pred_length,
				  num_channels = self.train_dataset.num_channels,
				  patch_len = int(config_dict['patch_len']),
				  stride = int(config_dict["stride"]),
				  num_prototypes = int(config_dict["num_prototypes"]),
				  support_set_size = int(config_dict["support_set_size"]),
				  top_k = int(config_dict["top_k"]),
				  num_llm_layers = int(config_dict["num_llm_layers"]),
				  learning_rate = float(config_dict["learning_rate"]),
				  batch_size = int(config_dict["batch_size"]),
				  lambda_weight = float(config_dict["lambda_weight"]),
				  weight_decay = float(config_dict["weight_decay"]),
				  )
			except ValueError:
			  print("ValueError -- try another set of distrib")
			  continue # If any inconstsitencey in params choices, iterate next turn


			# Incompatible avec la config init de HyperparameterConfig et les futut valeur set, risque de conflit qui doivent êter re-verifier a postriori
			#for key, value in config_dict.items():
			#	 setattr(base_config, key, value)

			# Re-verifier condition

			# Entraîner et évaluer
			try:
				result = self.train_and_evaluate(
					config = base_config,
					trial_id = trial_id,
					max_epochs = max_epochs,
					patience = patience,
					verbose = True
				)

				self.results.append(result)

				# Mettre à jour le meilleur
				if self.best_result is None or result.val_mse < self.best_result.val_mse:
					self.best_result = result
					print(f"\n🏆 Nouveau meilleur! Val MSE: {result.val_mse:.6f}")

			except Exception as e:
				print(f"❌ Trial {trial_id} échoué: {e}")
				continue

			# Sauvegarder (A CORRIGER ERRERU JSON)
			# self.save_results()

		# Trier
		self.results.sort(key=lambda x: x.val_mse)

		return self.results

	def save_results(self):
		"""Sauvegarde tous les résultats"""
		results_file = self.results_dir / 'all_results.json'

		results_data = {
			'results': [
				{
					'trial_id': r.trial_id,
					'config': json.dumps(r.config.to_dict()), # stringify dict
					'train_mse': r.train_mse,
					'val_mse': r.val_mse,
					'val_mae': r.val_mae,
					'train_time': r.train_time,
					'num_params': r.num_params,
					'timestamp': r.timestamp
				}
				for r in self.results
			],
			'best_result': {
				'trial_id': self.best_result.trial_id,
				'config': self.best_result.config.to_dict(),
				'val_mse': self.best_result.val_mse,
				'val_mae': self.best_result.val_mae
			} if self.best_result else None
		}

		with open(results_file, 'w') as f:
			json.dump(results_data, f, indent=2)

		print(f"\n✓ Résultats sauvegardés: {results_file}")

	def visualize_results(self, top_n: int = 10, save_path: Optional[str] = None):
		"""
		Visualise les résultats de la recherche

		Crée plusieurs graphiques pour analyser l'impact des hyperparamètres
		"""
		if not self.results:
			print("Aucun résultat à visualiser")
			return

		if save_path is None:
			save_path = self.results_dir / 'hyperparameter_analysis.png'

		fig = plt.figure(figsize=(18, 12))

		# 1. Top N configurations
		ax1 = plt.subplot(2, 3, 1)
		top_results = sorted(self.results, key=lambda x: x.val_mse)[:top_n]
		trial_ids = [r.trial_id for r in top_results]
		val_mses = [r.val_mse for r in top_results]

		ax1.barh(range(len(trial_ids)), val_mses, color='steelblue')
		ax1.set_yticks(range(len(trial_ids)))
		ax1.set_yticklabels([f'Trial {tid}' for tid in trial_ids])
		ax1.set_xlabel('Validation MSE')
		ax1.set_title(f'Top {top_n} Configurations')
		ax1.invert_yaxis()

		# 2. MSE vs MAE
		ax2 = plt.subplot(2, 3, 2)
		mses = [r.val_mse for r in self.results]
		maes = [r.val_mae for r in self.results]
		ax2.scatter(mses, maes, alpha=0.6, c=range(len(self.results)), cmap='viridis')
		ax2.set_xlabel('Validation MSE')
		ax2.set_ylabel('Validation MAE')
		ax2.set_title('MSE vs MAE')
		ax2.grid(True, alpha=0.3)

		# Marquer le meilleur
		if self.best_result:
			ax2.scatter([self.best_result.val_mse], [self.best_result.val_mae],
					   color='red', s=200, marker='*', edgecolors='black',
					   linewidths=2, label='Best', zorder=5)
			ax2.legend()

		# 3. Temps d'entraînement vs Performance
		ax3 = plt.subplot(2, 3, 3)
		times = [r.train_time for r in self.results]
		ax3.scatter(times, mses, alpha=0.6, c=range(len(self.results)), cmap='viridis')
		ax3.set_xlabel('Training Time (s)')
		ax3.set_ylabel('Validation MSE')
		ax3.set_title('Training Time vs Performance')
		ax3.grid(True, alpha=0.3)

		# 4. Impact du Learning Rate
		ax4 = plt.subplot(2, 3, 4)
		lrs = [r.config.learning_rate for r in self.results]
		ax4.scatter(lrs, mses, alpha=0.6)
		ax4.set_xlabel('Learning Rate')
		ax4.set_ylabel('Validation MSE')
		ax4.set_xscale('log')
		ax4.set_title('Learning Rate Impact')
		ax4.grid(True, alpha=0.3)

		# 5. Impact du Batch Size
		ax5 = plt.subplot(2, 3, 5)
		batch_sizes = [r.config.batch_size for r in self.results]
		unique_bs = sorted(set(batch_sizes))
		bs_mses = {bs: [] for bs in unique_bs}
		for r in self.results:
			bs_mses[r.config.batch_size].append(r.val_mse)

		bp_data = [bs_mses[bs] for bs in unique_bs]
		ax5.boxplot(bp_data, labels=unique_bs)
		ax5.set_xlabel('Batch Size')
		ax5.set_ylabel('Validation MSE')
		ax5.set_title('Batch Size Impact')
		ax5.grid(True, alpha=0.3, axis='y')

		# 6. Impact du Patch Length
		ax6 = plt.subplot(2, 3, 6)
		patch_lens = [r.config.patch_len for r in self.results]
		ax6.scatter(patch_lens, mses, alpha=0.6)
		ax6.set_xlabel('Patch Length')
		ax6.set_ylabel('Validation MSE')
		ax6.set_title('Patch Length Impact')
		ax6.grid(True, alpha=0.3)

		plt.suptitle('Hyperparameter Search Analysis', fontsize=16, y=0.995)
		plt.tight_layout()
		plt.savefig(save_path, dpi=150, bbox_inches='tight')
		print(f"\n✓ Visualisation sauvegardée: {save_path}")
		plt.close()

	def print_summary(self, top_n: int = 5):
		"""Affiche un résumé des résultats"""
		if not self.results:
			print("Aucun résultat disponible")
			return

		print(f"\n{'='*70}")
		print("RÉSUMÉ DE LA RECHERCHE D'HYPERPARAMÈTRES")
		print(f"{'='*70}")
		print(f"Nombre total de trials: {len(self.results)}")
		print(f"Temps total: {sum(r.train_time for r in self.results) / 60:.1f} minutes")

		if self.best_result:
			print(f"\n🏆 MEILLEUR RÉSULTAT (Trial {self.best_result.trial_id}):")
			print(f"  Val MSE: {self.best_result.val_mse:.6f}")
			print(f"  Val MAE: {self.best_result.val_mae:.6f}")
			print(f"  Temps d'entraînement: {self.best_result.train_time:.1f}s")
			print(f"  Paramètres: {self.best_result.num_params:,}")
			print(f"\n	Configuration:")
			for key, value in self.best_result.config.to_dict().items():
				print(f"	{key}: {value}")

		print(f"\n📊 TOP {top_n} CONFIGURATIONS:")
		top_results = sorted(self.results, key=lambda x: x.val_mse)[:top_n]

		for i, result in enumerate(top_results, 1):
			print(f"\n	{i}. Trial {result.trial_id}")
			print(f"	 Val MSE: {result.val_mse:.6f}, Val MAE: {result.val_mae:.6f}")
			print(f"	 LR: {result.config.learning_rate:.2e}, "
				  f"BS: {result.config.batch_size}, "
				  f"PL: {result.config.patch_len}, "
				  f"λ: {result.config.lambda_weight}")

#
# ===== FONCTION COMPLEMENTAIRES ==================
#

# ===== FONCTION UTILITAIRE POUR USAGE SIMPLE =====
def quick_hyperparameter_search(
	train_dataset,
	val_dataset,
	search_type: str = 'random',
	n_trials: int = 50,
	input_length: int = 10,
	pred_length: int = 1,
	num_channels: int = 1,
	device: str = 'cuda'
):
	"""
	Fonction simplifiée pour une recherche rapide

	Args:
		train_dataset: Dataset d'entraînement
		val_dataset: Dataset de validation
		search_type: 'random' ou 'grid'
		n_trials: Nombre de trials
		input_length: Longueur de l'input
		pred_length: Longueur de la prédiction
		num_channels: Nombre de canaux
		device: 'cuda' ou 'cpu'

	Returns:
		best_config, all_results, tuner
	"""
	print("="*70)
	print("RECHERCHE RAPIDE D'HYPERPARAMÈTRES")
	print("="*70)

	# Initialiser le tuner
	tuner = HyperparameterTuner(
		train_dataset=train_dataset,
		val_dataset=val_dataset,
		device=device,
		results_dir=f'./quick_search_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
		seed=42
	)

	if search_type == 'random':
		# Configuration par défaut pour random search
		param_distributions = {
			'input_length': [10, 20, 30, 50, 60],
			'learning_rate': (1e-3, 1e-2, 1e-1),  # Range log-uniform
			'batch_size': [4, 8, 16,],	 # Choix discrets (garder en dessous de 32 pour les cas "CUDA out of memory)
			'patch_len': (2, 4, 8, 10),				# Range uniforme
			'stride': (1, 2, 4, 5),
			'lambda_weight': (0.1, 5.0),
			'num_prototypes': [100, 500, 1000, 2000, 5000, 10000],
			'top_k': [4, 8, 12, 16],
			'num_llm_layers': [3, 6, 9],
			'weight_decay': (1e-5, 1e-3)}

		results = tuner.random_search(
			param_distributions=param_distributions,
			n_trials=n_trials,
			max_epochs=15,
			patience=5
		)

	elif search_type == 'grid':
		# Configuration par défaut pour grid search
		param_grid = {
			'learning_rate': [1e-4, 1e-3],
			'batch_size': [16, 32],
			'patch_len': [3, 5],
			'stride': [1, 2],
			'lambda_weight': [0.5, 1.0]
		}

		results = tuner.grid_search(
			param_grid=param_grid,
			max_epochs=15,
			patience=5
		)

	else:
		raise ValueError(f"search_type inconnu: {search_type}")

	# Afficher les résultats
	tuner.print_summary(top_n=5)
	tuner.visualize_results()

	return tuner.best_result.config, results, tuner


# ===== FONCTION POUR COMPARER PLUSIEURS CONFIGURATIONS =====

def compare_configurations(
	train_dataset,
	val_dataset,
	configs: List[HyperparameterConfig],
	device: str = 'cuda'
):
	"""
	Compare plusieurs configurations spécifiques

	Utile pour tester des hypothèses précises
	"""
	print("="*70)
	print("COMPARAISON DE CONFIGURATIONS")
	print("="*70)

	tuner = HyperparameterTuner(
		train_dataset=train_dataset,
		val_dataset=val_dataset,
		device=device,
		results_dir=f'./config_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
		seed=42
	)

	results = []
	for i, config in enumerate(configs, 1):
		print(f"\n{'='*70}")
		print(f"Configuration {i}/{len(configs)}")
		print(f"{'='*70}")

		result = tuner.train_and_evaluate(
			config=config,
			trial_id=i,
			max_epochs=30,
			patience=7,
			verbose=True
		)

		results.append(result)
		tuner.results.append(result)

		if tuner.best_result is None or result.val_mse < tuner.best_result.val_mse:
			tuner.best_result = result

	# Résumé
	tuner.print_summary(top_n=len(configs))

	return results, tuner


# ===== FONCTION POUR AFFINER AUTOUR D'UNE BONNE CONFIGURATION =====
def refine_search(
	channel_names,
	train_dataset,
	val_dataset,
	base_config: HyperparameterConfig,
	best_score: dict,
	n_trials: int = 30,
	device: str = 'cuda'
):
	"""
	Affine la recherche autour d'une configuration prometteuse

	Args:
		base_config: Configuration de base à affiner
		best_scores: Meilleurs score obtenu précédement
		  {
			"mse": (int),
			"mae": (int)
		  }
		n_trials: Nombre de variations à tester
	"""
	print("="*70)
	print("AFFINEMENT AUTOUR D'UNE CONFIGURATION")
	print("="*70)
	print("\nConfiguration de base:")
	for key, value in base_config.to_dict().items():
		print(f"  {key}: {value}")

	tuner = HyperparameterTuner(
		data = df,
		train_dataset=train_dataset,
		val_dataset=val_dataset,
		channel_names = channel_names,
		device=device,
		results_dir=f'./refinement_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
		seed=42
	)

	# Créer des variations autour de la config de base
	# Varier ±20% pour les valeurs continues, ±1 pour les discrètes
	for trial_id in range(1, n_trials + 1):
		new_config = HyperparameterConfig(**base_config.to_dict())

		# Varier learning rate (±50% en échelle log)
		lr_factor = np.random.uniform(0.5, 2.0)
		new_config.learning_rate = base_config.learning_rate * lr_factor

		# Varier batch size (±1 niveau)
		bs_options = [8, 16, 32, 64]
		current_idx = bs_options.index(base_config.batch_size) if base_config.batch_size in bs_options else 1
		new_idx = max(0, min(len(bs_options) - 1, current_idx + np.random.randint(-1, 2)))
		new_config.batch_size = bs_options[new_idx]

		# Varier patch_len (±1)
		new_config.patch_len = max(2, min(
			base_config.input_length - 1,
			base_config.patch_len + np.random.randint(-1, 2)
		))

		# Varier stride (±1)
		new_config.stride = max(1, base_config.stride + np.random.randint(-1, 2))

		# Varier lambda_weight (±30%)
		new_config.lambda_weight = base_config.lambda_weight * np.random.uniform(0.7, 1.3)

		# Entraîner
		try:
			result = tuner.train_and_evaluate(
				config=new_config,
				trial_id=trial_id,
				max_epochs=25,
				patience=7,
				verbose=True
			)

			tuner.results.append(result)

			#tuner.best_result.val_mse < best_score["mse"]
			#if tuner.best_result is None or result.val_mse < tuner.best_result.val_mse:
			if tuner.best_result is None or (tuner.best_result.val_mse < best_score["mse"] and tuner.best_result.val_mae <= best_score["mae"]):
				tuner.best_result = result
				print(f"\n🎯 Amélioration trouvée! Val MSE: {result.val_mse:.6f}")

		except Exception as e:
			print(f"❌ Trial {trial_id} échoué: {e}")
			continue

	tuner.print_summary(top_n=10)
	tuner.visualize_results()

	return tuner.best_result.config, tuner.results, tuner


# ===== EXPORT DES RÉSULTATS EN CSV =====
def export_results_to_csv(results: List[TrialResult], output_path: str = 'results.csv'):
	"""Exporte les résultats en CSV pour analyse externe"""
	import csv

	with open(output_path, 'w', newline='') as f:
		writer = csv.writer(f)

		# Header
		header = ['trial_id', 'val_mse', 'val_mae', 'train_mse', 'train_time', 'num_params']
		header.extend(list(results[0].config.to_dict().keys()))
		writer.writerow(header)

		# Données
		for r in results:
			row = [
				r.trial_id, r.val_mse, r.val_mae, r.train_mse,
				r.train_time, r.num_params
			]
			row.extend(list(r.config.to_dict().values()))
			writer.writerow(row)

	print(f"✓ Résultats exportés: {output_path}")


# ===== GUIDE D'UTILISATION =====
"""
HYPERPARAMÈTRES À PRIORISER:

1. learning_rate: Impact majeur, toujours à optimiser
2. batch_size: Équilibre vitesse/qualité
3. patch_len et stride: Crucial pour capturer les patterns
4. lambda_weight: Balance entre forecast et contrastive losses
5. num_prototypes et top_k: Moins critique, optimiser en second

TEMPS ESTIMÉS (pour référence):

- Grid Search (3^5 = 243 combos): 8-12 heures
- Random Search (100 trials): 4-6 heures
- Refinement (30 trials): 1-2 heures
- Quick search (50 trials): 2-3 heures

Ces temps dépendent de:
- Taille du dataset
- input_length et pred_length
- Hardware (GPU/CPU)
- max_epochs et patience
"""