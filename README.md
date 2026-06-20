# EFU: Enforcing Federated Unlearning via Functional Encryption

📄 Paper: [CIKM '25](https://doi.org/10.1145/3746252.3761091) · [arXiv:2508.07873](https://arxiv.org/abs/2508.07873)

Federated unlearning lets clients exercise their "right to be forgotten" by
removing the influence of their data from a collaboratively trained model.
Existing methods unlearn locally and send targeted updates to the server, but
rely on server-side cooperation — revealing the client's intent and identity
without enforcement. EFU (Enforced Federated Unlearning) is a cryptographically
enforced framework that lets clients initiate unlearning while concealing its
occurrence from the server. EFU leverages functional encryption to bind encrypted
updates to a specific aggregation function, so the server can neither perform
unauthorized computations nor detect or skip unlearning requests. To further mask
behavioral and parameter shifts in the aggregated model, EFU adds auxiliary
unlearning losses based on adversarial examples and parameter-importance
regularization. EFU achieves near-random accuracy on forgotten data while
maintaining performance comparable to full retraining across datasets and
architectures — all while concealing unlearning intent — and is agnostic to the
underlying unlearning algorithm.

## Methods

- **`src/normal.py`** — standard federated learning (`FL`), optionally with
  differential privacy (`FL+DP`).
- **`src/unlearn.py`** — federated *unlearning* (`FU`): from `--erase_rnd`, the
  clients in `--erase_cids` unlearn and all updates are aggregated under
  functional encryption, so learning and unlearning are indistinguishable to the
  server. The client-side unlearning algorithm is selected with
  `--unlearn_method`:
    - `efu` — EFU's 3-term loss (forget + adversarial feature-collapse + MAS
      parameter-drift regularization).
    - `pgd` — Projected Gradient Ascent (Halimi et al., 2022).
    - `sga_ewc` — Stochastic Gradient Ascent + EWC (Wu et al., 2022).
    - `fedosd` — FedOSD core (Pan et al., 2025).
  EFU is agnostic to this choice, so any baseline can be run plain or wrapped by
  the FE aggregation (the paper's `EFU_PGD`, `EFU_SGA-EWC` rows).
- **`naive_unlearn/`** — Full Retrain baseline (retrain without the erased data).

Datasets (`--ds`): `cifar10`, `lfw`, `timagenet`. Models (`--model_name`):
`vgg16`, `resnet18`, `convnet`, `lenet`, `fc`.

## Setup

```bash
bash install.sh                  # installs the DMCFE backend (modules/mife)
pip install -r requirements.txt
```

## Usage

```bash
cd src
# standard federated training
python normal.py --ds cifar10 --model_name vgg16 --num_clients 5 --num_rounds 150

# enforced unlearning: erase client 1 starting at round 90
python unlearn.py --ds cifar10 --model_name vgg16 --num_clients 5 \
    --num_rounds 150 --erase_cids 1 --erase_rnd 90

# a comparison baseline instead of EFU's own unlearning loss
python unlearn.py --ds lfw --model_name resnet18 --erase_cids 1 --unlearn_method pgd
```

The functional-encryption aggregation decrypts via a bounded discrete-log search,
so it suits compact models / cluster counts; `--kappa` controls the number of
encrypted centroids.

## Citation

```bibtex
@inproceedings{mohammadi2025efu,
  title={EFU: Enforcing Federated Unlearning via Functional Encryption},
  author={Mohammadi, Samaneh and Tsouvalas, Vasileios and Symeonidis, Iraklis and Balador, Ali and Ozcelebi, Tanir and Flammini, Francesco and Meratnia, Nirvana},
  booktitle={Proceedings of the 34th ACM International Conference on Information and Knowledge Management (CIKM '25)},
  year={2025},
  doi={10.1145/3746252.3761091}
}
```

## License

MIT. The vendored `mife` (PyMIFE) library retains its own license.
