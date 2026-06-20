import torch
import torchmetrics
import numpy as np
import copy
from data import load_data

class Network(torch.nn.Module):

	def __init__(self, num_classes=10):
		super().__init__()
		self.num_classes = num_classes
		self.optimizer, self.metrics = None, None
		self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)
		self.conv2 = torch.nn.Conv2d(16, 32, 3, padding=1)
		self.fc1 = torch.nn.Linear(32 * 8 * 8, 128)
		self.fc2 = torch.nn.Linear(128, num_classes)

	def forward(self, x):
		x = torch.nn.functional.relu(self.conv1(x))
		x = torch.nn.functional.max_pool2d(x, 2)
		x = torch.nn.functional.relu(self.conv2(x))
		x = torch.nn.functional.max_pool2d(x, 2)
		x = x.view(x.size(0), -1)
		x = torch.nn.functional.relu(self.fc1(x))
		x = self.fc2(x)
		return x

	@property
	def device(self):
		return next(self.parameters()).device

	def load_state_dict(self, state_dict):
		if self.device != next(iter(state_dict.values())).device:
			state_dict = {k: v.to(self.device) for k, v in state_dict.items()}
		super().load_state_dict(state_dict)

	def get_state_dict(self, device='cpu'):
		state_dict = self.state_dict()
		if device != self.device:
			return {k: v.to(device) for k, v in state_dict.items()}
		return copy.deepcopy(state_dict)

	def compile(self, lr=1e-3, ):
		if self.optimizer is None:
			self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
		if self.metrics is None:
			self.metrics = torchmetrics.Accuracy(task='multiclass', num_classes=self.num_classes).to(self.device)

	def _train(self, ds, epochs, cid='N/A', lr=1e-3, verbose=True):
		self.compile(lr=lr)
		#self.metrics.reset()
		self.train()
		history = {'loss':[], 'acc': []}
		for e in range(epochs):
			self.metrics.reset()
			epoch_loss = 0.
			for sample in ds:
				x, y = sample['image'].to(self.device, non_blocking=True), sample['label'].to(self.device, non_blocking=True).long()
				self.optimizer.zero_grad()
				logits = self(x)
				y_prob = torch.nn.functional.log_softmax(logits, dim=1)
				_loss = torch.nn.functional.nll_loss(y_prob, y)
				_loss.backward()
				self.optimizer.step()
				epoch_loss += _loss.item()
				y_preds = torch.argmax(y_prob, dim=1)
				self.metrics(y_preds, y)
			history['loss'].append(epoch_loss/len(ds))
			history['acc'].append(self.metrics.compute().item())
			if verbose:
				print(f"[Client {cid+1}] Epoch {e+1}/{epochs} - Loss: {history['loss'][-1]:.4f} - Acc: {100.*history['acc'][-1]:.2f}")
		return history

	def _test(self, ds, verbose=True):
		if self.metrics is None:
			self.metrics = torchmetrics.Accuracy(task='multiclass', num_classes=self.num_classes).to(self.device)
		self.metrics.reset()
		self.eval()
		test_loss = 0.0
		with torch.no_grad():
			for sample in ds:
				x, y = sample['image'].to(self.device, non_blocking=True), sample['label'].to(self.device, non_blocking=True).long()
				logits = self(x)
				y_prob = torch.nn.functional.log_softmax(logits, dim=1)
				_loss = torch.nn.functional.nll_loss(y_prob, y)
				test_loss += _loss.item()
				y_preds = torch.argmax(y_prob, dim=1)
				self.metrics(y_preds, y)
		test_loss /= len(ds)
		test_acc = self.metrics.compute().item()
		if verbose:
			print(f"Test - Loss: {test_loss:.4f} - Acc: {100. * test_acc:.2f}%")
		return {'loss': test_loss, 'accuracy': test_acc}

class Server:

	def __init__(self, data, num_clients, batch_size=128, device='cuda:0', seed=42, **kwargs):
		self.device = device
		self.num_clients = num_clients
		self.data = torch.utils.data.DataLoader(dataset=data, batch_size=batch_size, shuffle=False, 
			pin_memory=True, pin_memory_device=self.device,
			num_workers=1, persistent_workers=True, worker_init_fn=lambda _: np.random.seed(seed))
		self.model = Network(**kwargs).to(self.device)

	def aggregation(self, model_updates):
		model_updates = {cid: update for cid,(_, update) in model_updates.items()}
		with torch.no_grad():
			param_vectors = [torch.nn.utils.parameters_to_vector(model_updates[cid].values()) for cid in model_updates]
			sum_vec = torch.stack(param_vectors).mean(dim=0)
			torch.nn.utils.vector_to_parameters(sum_vec, self.model.parameters())
		return self.model.state_dict()

	def weighted_aggregation(self, model_updates):
		with torch.no_grad():
			total_samples = sum(num_samples for num_samples, _ in model_updates.values())
			weighted_sum = None
			for num_samples, state_dict in model_updates.values():
				param_vec = torch.nn.utils.parameters_to_vector(state_dict.values())
				weight = num_samples / total_samples
				weighted_sum = param_vec*weight if weighted_sum is None else  weighted_sum+param_vec*weight
			torch.nn.utils.vector_to_parameters(weighted_sum, self.model.parameters())
		return self.model.state_dict()

	def evaluate(self, verbose=False):
		return self.model._test(ds=self.data, verbose=verbose)

	def compute_distance_from_server_model(self, model_updates, device='cpu', verbose=True):
		_device = next(self.model.parameters()).device
		if _device != torch.device(device):
			self.model.to(device)
		global_flat = torch.nn.utils.parameters_to_vector(self.model.parameters())
		model_updates = list(model_updates.items())  # Convert dictionary to list of tuples
		distances = {}
		with torch.no_grad():
			for cid, (_, state_dict) in model_updates:
				state_dict = {k: v.to(device) if v.device != torch.device(device) else v for k, v in state_dict.items()}
				temp_model = self.model.__class__().to(device)
				temp_model.load_state_dict(state_dict)
				update_flat = torch.nn.utils.parameters_to_vector(temp_model.parameters())
				distances[cid] = torch.square(torch.norm(global_flat - update_flat)).item()
				if verbose:
					print(f"[Server] Distance from Client {cid+1}: {distances[cid]}")
		
		if _device != torch.device(device):
			self.model.to(_device)

class Client:

	def __init__(self, cid, data, batch_size=128, device='cuda:0', lr=1e-3, **kwargs):
		self.cid = cid
		self.device = device
		self.num_samples = len(data)
		self.data = torch.utils.data.DataLoader(dataset=data, batch_size=batch_size, shuffle=True, \
			pin_memory=True, pin_memory_device=self.device,
			num_workers=1, persistent_workers=True, worker_init_fn=lambda _: np.random.seed(self.cid))
		self.model = Network(**kwargs).to(self.device)
		self.lr = lr

	def local_epoch(self, server_update, epochs=1, unlearn=False, verbose=False):

		device = next(iter(server_update.values())).device
		if not unlearn:
			self.model.load_state_dict(server_update)

		if unlearn:
			model_update = self.compute_unlearn_update(server_update, device=device, verbose=True)
			train_metrics = {'loss': [0.0], 'acc': [0.0]}
		else:
			train_metrics = self.model._train(cid=self.cid, ds=self.data, epochs=epochs, verbose=verbose)
			model_update = self.model.get_state_dict(device=device)

		return train_metrics, (self.num_samples,model_update)

	def compute_unlearn_update(self, server_update, unlearn_rate=2., device='cpu', verbose=True):
		if verbose:
			print(f"[Client {self.cid+1}] Initiate unlearning process to remove his data from server.")
		server_state_dict = {k: v.to(device) if v.device != torch.device(device) else v for k, v in server_update.items()}
		client_state_dict = self.model.get_state_dict(device=device)
		return {k: unlearn_rate * server_state_dict[k] - client_state_dict[k] for k in client_state_dict}


if __name__ == "__main__":

	num_clients = 5
	train_batch_size = 128
	test_batch_size = 1024
	device = 'cuda:1'
	num_rounds = 10
	unlearn_cid, unlearn_rnd = 0, 2

	# Prepare FL setup
	dataset = load_data('cifar10', num_partitions=num_clients, split='iid')
	server = Server(data=dataset.load_test_set(), num_clients=num_clients, batch_size=test_batch_size, device=device, num_classes=dataset.num_classes)
	clients = [Client(cid=cid, data=dataset.load_partition(cid), batch_size=32, device=device, num_classes=dataset.num_classes) for cid in range(num_clients)]
	# Federated Learning Simulation
	history, model_updates = {}, {}
	for rnd in range(num_rounds):
		print(f"\nRound {rnd+1}/{num_rounds}")
		# Step 1: Get the current server model state and move it to CPU
		server_state_dict = server.model.get_state_dict(device='cpu')
		# Step 2: Client-side operations
		for cid in range(num_clients):
			# Train local model for one epoch
			history[cid], model_updates[cid] = clients[cid].local_epoch(server_update=server_state_dict,
				unlearn=True if (cid==unlearn_cid and rnd>=unlearn_rnd) else False, epochs=1)
			# Print training progress for the client
			print(f"[Client {cid+1}] - train_loss: {history[cid]['loss'][-1]:.4f}, train_acc: {100.*history[cid]['acc'][-1]:.2f}%")
		# Step 3: Aggregate client model updates on the server
		server.weighted_aggregation(model_updates)
		_ = server.compute_distance_from_server_model(model_updates, device='cpu', verbose=True)
		# Step 4: Evaluate the updated server model
		metrics = server.evaluate()
		print(f"[Server] - test_loss: {metrics['loss']:.4f}, test_acc: {100.*metrics['accuracy']:.2f}%")
