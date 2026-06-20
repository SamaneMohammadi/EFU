import os
os.environ["HF_DATASETS_CACHE"] = '../datasets'
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
import copy
import torch
import inspect
import numpy as np
# import transformers
# import torchmetrics

from networks.network import networks, processors
from dataloaders.data import dataloaders
from utils.tools import logger_fn, euclidean_distance
from aggregate import fedavg_aggr as aggregation_fn
from client import local_train
from args import get_args

DATA_DIR = '../datasets'


class FederatedTraining:

	def __init__(self, data_fn, model_fn, processor_fn, num_clients, data_args=None, model_args=None, num_rounds=5, local_epochs=1, lr=3e-4, ckpt_dir='../assets/', store_progress=True, device='cuda', **kwargs):
		self.num_clients = num_clients
		self.data_args = data_args or {}
		self.model_args = model_args or {}
		self.data_fn = data_fn
		self.model_fn = model_fn
		self.processor_fn = processor_fn
		self.model = self.model_fn(**self.model_args)
		self.train_ds, self.test_ds, self.num_classes = self.data_fn(num_clients=self.num_clients, processor_fn=self.processor_fn, **self.data_args)
		self.model = self.model.set_clf(self.num_classes).to("cpu")
		self.num_rounds, self.local_epochs, self.lr = num_rounds, local_epochs, lr
		self.clients_weights = [None for _ in range(num_clients)]
		self.ds_name = self.test_ds.dataset.name
		self.store_progress = store_progress
		self.ckpt_dir = ckpt_dir
		self.cur_round = 0
		self.best_acc = 0.0
		self.train_metrics = {}
		self.device = device
		self.logger = None
		self.m_t, self.v_t = None, None

	def train(self, exclude_cids = []):
		self._init_train()
		###################################################################################################
		global_weights = self.model.get_weights()
		lr = max(self.lr/np.sqrt(self.cur_round) if self.cur_round>0 else self.lr, 3e-5)
		for r in range(self.cur_round, self.num_rounds):
			self.logger.info(f"\n Round {r+1} initiated. (lr: {lr:.5f})")
			# Client Operation
			cids, cids_weights, cids_num_samples, cids_metrics = list((set(range(self.num_clients))) - set(exclude_cids)), [], [], []
			for cid in cids:
				cid_model = self.model_fn(num_classes=self.num_classes, **self.model_args).set_weights(global_weights)
				cid_model, cid_samples, metrics = local_train(model=cid_model.to(self.device), data=self.train_ds[cid], lr=lr, num_epochs=self.local_epochs)
				cids_weights.append(cid_model.to("cpu").get_weights()); cids_num_samples.append(cid_samples); cids_metrics.append(metrics)
				self.clients_weights[cid] = copy.deepcopy(cids_weights[-1])
				self.logger.info(f"[Client {cid+1}] - Loss: {metrics['loss']:.4f}, Acc: {metrics['accuracy']:.2%}")

			# Server Operation
			global_weights, self.m_t, self.v_t, _ = aggregation_fn(
				cids_weights, cids_num_samples, rnd=self.cur_round+1, model=self.model, eta=self.lr, m_t=self.m_t, v_t=self.v_t)
			self._clients_distance_from_server(global_weights)
			self._update_progress(self.evaluate(), cids_metrics)
		###################################################################################################

	def evaluate(self):
		self.model.to(self.device)
		metrics = self.model.evaluate(self.test_ds)
		self.model.to("cpu")
		self.logger.info(f"[Server] - Loss: {metrics['loss']:.4f}, Acc: {metrics['accuracy']:.2%}")
		return metrics

	def save_ckpt(self, ckpt_fp):
		snapshot = {key: getattr(self, key) for key in self.__dict__.keys() if key != "model"}
		if hasattr(self, "model") and self.model is not None:
			snapshot["model_state_dict"] = self.model.get_weights()
		torch.save(snapshot, ckpt_fp)
		self.logger.info(f"Checkpoint saved at {ckpt_fp}")

	def load_ckpt(self, ckpt_fp):
		_device = self.device
		_store_progress = self.store_progress
		checkpoint = torch.load(ckpt_fp, map_location="cpu", weights_only=False)

		# Restore all attributes except model
		for key, value in checkpoint.items():
			if key != "model_state_dict":
				setattr(self, key, value)

		# Reload the model and apply state_dict
		if "model_state_dict" in checkpoint:
			self.model = self.model_fn(num_classes=self.num_classes, **self.model_args).to("cpu")  # Reinitialize model
			self.model.set_weights(checkpoint["model_state_dict"])  # Load weights

		self.device = _device
		self.store_progress = _store_progress

		self._reload_data()  # To fix randomness issues
		self.logger.info(f"Checkpoint loaded from {ckpt_fp}.\n\n")
		self._print_cfg()  # Print setup

	def _reload_data(self):
		self.train_ds, self.test_ds, self.num_classes = self.data_fn(num_clients=self.num_clients, processor_fn=self.processor_fn, **self.data_args)

	def _print_cfg(self):
		_train_type = ("FU+DP" if getattr(self, "max_grad", None) is not None and getattr(self, "erase_cids", None) is not None else
			"FU" if getattr(self, "max_grad", None) is None and getattr(self, "erase_cids", None) is not None else
			"FL+DP" if getattr(self, "max_grad", None) is not None and getattr(self, "erase_cids", None) is None else "FL")
		config_dict = {
			"Num clients": getattr(self, "num_clients", None), "Lora rank": self.model_args.get("lora_rank", None), "Erase cids": getattr(self, "erase_cids", None),
			"Lr": getattr(self, "lr", None), "Delta": getattr(self, "delta", None), "Max grad": getattr(self, "max_grad", None), "Max noise": getattr(self, "max_noise", None),
			"Num rounds": getattr(self, "num_rounds", None), "Local epochs": getattr(self, "local_epochs", None), "Erase rnd": getattr(self, "erase_rnd", None),
			"Batch size": self.data_args.get("batch_size", None), "Seed": self.data_args.get("seed", None), "Device": getattr(self, "device", None),
			"Train type": _train_type, "Verbose": getattr(self, "verbose", False), "Ckpt dir": getattr(self, "ckpt_dir", None),}
		self.logger.info("=" * 50)
		self.logger.info("Experiment Configuration")
		self.logger.info("=" * 50)
		for key, value in config_dict.items():
			self.logger.info(f"{key:<20}: {value}")
		self.logger.info("=" * 50)

	def _clients_distance_from_server(self, global_weights):
		distances = [euclidean_distance(global_weights, w) if w is not None else None for w in self.clients_weights]
		erase_cids = getattr(self, "erase_cids", [])
		distances = [(i, d) for i, d in enumerate(distances) if d is not None]
		inner = np.mean(inner_values) if (inner_values := [d for i, d in distances if i not in erase_cids]) else "N/A"
		outer = np.mean(outer_values) if (outer_values := [d for i, d in distances if i in erase_cids]) else "N/A"
		#self.logger.info(f"[Server] - Avg. Distance to train models: {inner:.4f}, Avg. Distance to unlearn models: {outer if isinstance(outer, str) else f'{outer:.4f}'}")
		self.logger.info(f"[Server] - \033[92mLearned Model Distance: {inner:.4f}\033[0m | " + f"\033[91mUnlearned Model Distance: {outer if isinstance(outer, str) else f'{outer:.4f}'}\033[0m")

	def _update_progress(self, eval_metrics, cids_metrics=None):
		self.train_metrics[self.cur_round] = {'server':eval_metrics, 'clients': cids_metrics}
		self.cur_round+=1
		if eval_metrics['accuracy'] > self.best_acc:
			self.best_acc = eval_metrics['accuracy']
			if self.store_progress:
				self.save_ckpt(ckpt_fp=os.path.join(self.ckpt_dir,f"{self.ds_name}_{self.num_clients}_{self.num_rounds}_{self.cur_round}_{self.best_acc:.2f}.tar"))
				self.logger.info(f"New best model saved with accuracy: {self.best_acc:.2%}.")

	def _init_train(self):
		self.logger = logger_fn(self.ckpt_dir if self.store_progress else None)
		self._print_cfg()

if __name__ == "__main__":
	args = get_args()
	fl = FederatedTraining(
		data_fn=dataloaders[args.ds], 			data_args  = {'root': os.path.join(DATA_DIR, args.ds), 'batch_size': args.batch_size, 'seed': args.seed},
		model_fn=networks[args.model_name],		model_args = {'model_name': args.model_name, 'lora_rank': args.lora_rank, 'hidden_dim': 512, 'input_dim': 3072},
		processor_fn=processors[args.model_name],
		store_progress=True,
		**{k: v for k, v in vars(args).items() if k in inspect.signature(FederatedTraining.__init__).parameters})
	fl.train(exclude_cids=[])