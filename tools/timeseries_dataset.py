import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

"""
  _____ _                  __           _                  ___      _                 _   
 /__   (_)_ __ ___   ___   / _\ ___ _ __(_) ___  ___      /   \__ _| |_ __ _ ___  ___| |_ 
   / /\/ | '_ ` _ \ / _ \  \ \ / _ \ '__| |/ _ \/ __|    / /\ / _` | __/ _` / __|/ _ \ __|
  / /  | | | | | | |  __/  _\ \  __/ |  | |  __/\__ \   / /_// (_| | || (_| \__ \  __/ |_ 
  \/   |_|_| |_| |_|\___|  \__/\___|_|  |_|\___||___/  /___,' \__,_|\__\__,_|___/\___|\__|
                                                                                       ...
"""
class TimeSeriesDataset(Dataset):

	"""
        Classe pour la représentation de séries temporelles multivariées
    """

	def __init__(
		self,
		data: np.ndarray,
		input_length: int = 512,
		pred_length: int = 96
	):
		"""
		Args:
			data: (T, C) ou (T,) pour univarié
		"""
		if data.ndim == 1:
			data = data.reshape(-1, 1)

		self.data = torch.FloatTensor(data)	 # (C, T)
		self.input_length = input_length
		self.pred_length = pred_length
		self.num_channels = data.shape[1]

	def __len__(self):
		return len(self.data) - self.input_length - self.pred_length + 1

	def __getitem__(self, idx):
		x = self.data[idx:idx + self.input_length].T  # (C, T)
		#y = self.data[idx + self.input_length:idx + self.input_length + self.pred_length].T  # (C, H)
		y = self.data[idx + self.input_length].T  # (, H)

		return x, y
