import os
os.environ["HF_DATASETS_CACHE"] = '../datasets'
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
import copy
import inspect

import numpy as np

from networks.network import networks, processors
from dataloaders.data import dataloaders
from aggregate import fe_aggr
from client import local_train, local_unlearn
from utils.baselines import pgd_halimi, sga_ewc, fedosd
from args import get_args
from normal import FederatedTraining

UNLEARN_METHODS = {"efu": local_unlearn, "pgd": pgd_halimi, "sga_ewc": sga_ewc, "fedosd": fedosd}

DATA_DIR = '../datasets'


class FederatedUTraining(FederatedTraining):
	"""Federated training with EFU unlearning.

	From `erase_rnd`, the clients in `erase_cids` switch from local training to
	local *unlearning* (the EFU 3-term loss) for `erase_duration` rounds, then
	leave. All updates are aggregated under functional encryption (fe_aggr), so
	the server cannot tell learning from unlearning.
	"""

	def __init__(self, erase_cids=[], erase_rnd=90, erase_duration=10, kappa=64, unlearn_method="efu", **kwargs):
		super().__init__(**kwargs)
		self.erase_cids = erase_cids
		self.erase_rnd = erase_rnd
		self.erase_duration = erase_duration
		self.kappa = kappa
		self.unlearn_method = unlearn_method
		self.erase_labels = sorted({l for cid in erase_cids
			for l in self.train_ds[cid].dataset.labels})

	def train(self, exclude_cids=[]):
		self._init_train()
		global_weights = self.model.get_weights()
		lr = self.lr / np.sqrt(self.cur_round) if self.cur_round > 0 else self.lr
		for r in range(self.cur_round, self.num_rounds):
			self.logger.info(f"\n Round {r + 1} initiated. (lr: {lr:.5f})")
			cids = list(set(range(self.num_clients)) - set(exclude_cids))
			cids_weights, cids_num_samples, cids_metrics = [], [], []
			for cid in cids:
				cid_model = self.model_fn(num_classes=self.num_classes, **self.model_args)
				cid_model = cid_model.set_clf(self.num_classes, label2idx=self.model.label2class)
				cid_model = cid_model.set_weights(global_weights).to(self.device)

				unlearn = (r + 1 >= self.erase_rnd) and (cid in self.erase_cids)
				if unlearn:
					cid_model, n, metrics = UNLEARN_METHODS[self.unlearn_method](cid_model, self.train_ds[cid], lr=2e-4, num_epochs=1)
					if r + 1 == self.erase_rnd + self.erase_duration - 1:   # last unlearning round
						exclude_cids.append(cid)
				else:
					cid_model, n, metrics = local_train(cid_model, self.train_ds[cid], lr=lr, num_epochs=self.local_epochs)

				cids_weights.append(cid_model.to("cpu").get_weights())
				cids_num_samples.append(n)
				cids_metrics.append(metrics)
				if not unlearn:
					self.clients_weights[cid] = copy.deepcopy(cids_weights[-1])
				self.logger.info(f"[Client {cid + 1}] - Loss: {metrics['loss']:.4f}, Acc: {metrics['accuracy']:.2%}"
					+ ("  [unlearning]" if unlearn else ""))

			# functional-encryption secure aggregation (learn/unlearn indistinguishable)
			global_weights, self.m_t, self.v_t, lr = fe_aggr(
				cids_weights, cids_num_samples, rnd=self.cur_round + 1, eta=self.lr,
				model=self.model, kappa=self.kappa)
			self._clients_distance_from_server(global_weights)
			self._update_progress(self.evaluate(), cids_metrics)

	def evaluate(self):
		self.model.to(self.device)
		metrics = self.model.evaluate_classes(self.test_ds, erased_classes=self.erase_labels)
		self.model.to("cpu")
		self.logger.info(
			f"\033[91mErased: {100. * metrics['erase_acc']:.2f}%\033[0m | "
			f"\033[92mNon-Erased: {100. * metrics['n_erase_acc']:.2f}%\033[0m | "
			f"\033[94mTotal: {metrics['accuracy']:.2f}%\033[0m")
		return metrics

	def _update_progress(self, eval_metrics, cids_metrics=None):
		self.train_metrics[self.cur_round] = {'server': eval_metrics, 'clients': cids_metrics}
		self.cur_round += 1
		if eval_metrics['n_erase_acc'] > self.best_acc:
			self.best_acc = eval_metrics['n_erase_acc']
			if self.store_progress:
				self.save_ckpt(ckpt_fp=os.path.join(
					self.ckpt_dir, f"{self.ds_name}_{self.num_clients}_{self.num_rounds}_{self.cur_round}_{self.best_acc:.2f}.tar"))
				self.logger.info(f"New best model saved with accuracy: {self.best_acc:.2%}.")


if __name__ == "__main__":
	args = get_args()
	fl = FederatedUTraining(
		data_fn=dataloaders[args.ds],
		data_args={'root': os.path.join(DATA_DIR, args.ds), 'batch_size': args.batch_size, 'seed': args.seed},
		model_fn=networks[args.model_name],
		model_args={'model_name': args.model_name, 'lora_rank': args.lora_rank, 'hidden_dim': 512, 'input_dim': 3072},
		processor_fn=processors[args.model_name],
		store_progress=True,
		**{k: v for k, v in vars(args).items()
			if k in {**inspect.signature(FederatedUTraining.__init__).parameters,
				**inspect.signature(FederatedTraining.__init__).parameters}})
	fl.train(exclude_cids=[])
