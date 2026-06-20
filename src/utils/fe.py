"""Functional-encryption secure aggregation for EFU.

EFU binds each client's (learn or unlearn) update to a fixed aggregation
function via decentralized multi-client functional encryption (DMCFE, the
EncCluster/PMC scheme), so the server can only compute the agreed weighted
average over ciphertexts and cannot tell a learning update from an unlearning
one. To keep encryption cheap, weights are first compressed by clustering into a
small set of centroids (only the centroids are encrypted); the cluster-weight
mapping is sent alongside.

This module provides the crypto core used by `aggregate.fe_aggr`:
  * cluster a client's weights into `kappa` centroids + mapping,
  * encrypt the centroids under each client's key,
  * decrypt the weighted inner product  sum_n |D_n| * Z_n[P_n[i]]  / |D|
    (weighted FedAvg) without revealing any individual update.

The scheme recovers results by a bounded discrete-log search, so the fixed-point
precision is sized to the total sample count, and BatchNorm buffers (not weights)
are averaged in plaintext rather than encrypted.
"""

import math

import numpy as np
import torch
from sklearn.cluster import KMeans
from mife.multiclient.ddh import FeDDHMultiClient

BOUND = (-10_000_000, 10_000_000)
_MAX_PREC = 5
_BUFFERS = ("running_mean", "running_var", "num_batches_tracked")


def _is_weight(name):
    return not any(b in name for b in _BUFFERS)


def _flatten(state):
    keys, shapes, chunks = [], [], []
    for k, v in state.items():
        if _is_weight(k):
            keys.append(k); shapes.append(tuple(v.shape))
            chunks.append(v.detach().cpu().numpy().ravel())
    return np.concatenate(chunks).astype(np.float64), keys, shapes


def _unflatten(vec, keys, shapes):
    out, i = {}, 0
    for k, shape in zip(keys, shapes):
        n = int(np.prod(shape)) if shape else 1
        out[k] = torch.tensor(vec[i:i + n].reshape(shape), dtype=torch.float32); i += n
    return out


def cluster(state, kappa, seed=0):
    """Compress weights into (centroids Z[kappa], mapping P[d], keys, shapes)."""
    vec, keys, shapes = _flatten(state)
    km = KMeans(n_clusters=kappa, n_init=4, random_state=seed)
    P = km.fit_predict(vec.reshape(-1, 1))
    return km.cluster_centers_.ravel().astype(np.float64), P.astype(np.int64), keys, shapes


class DMCFE:
    """Decentralized multi-client FE for weighted secure aggregation."""

    def __init__(self, num_clients, total_samples, bound=BOUND):
        self.n = num_clients
        self.bound = bound
        self.key = FeDDHMultiClient.generate(num_clients, 1)
        head = bound[1] / max(total_samples * 8.0, 1.0)
        self.prec = max(0, min(_MAX_PREC, int(math.log10(max(head, 1)))))

    def encrypt_centroids(self, cid, centroids, tag):
        ek = self.key.get_enc_key(cid)
        s = 10 ** self.prec
        return [FeDDHMultiClient.encrypt([int(round(float(z) * s))], tag, ek) for z in centroids]

    def secure_aggregate(self, ciphertexts, mappings, num_examples, tag, d):
        pk = self.key.get_public_key()
        sk = FeDDHMultiClient.keygen([[int(num_examples[i])] for i in range(self.n)], self.key)
        total, s = sum(num_examples), 10 ** self.prec
        mappings = [np.asarray(P, dtype=int) for P in mappings]
        out, cache = np.empty(d), {}
        for i in range(d):
            tup = tuple(int(P[i]) for P in mappings)
            if tup not in cache:
                cols = [ciphertexts[n][tup[n]] for n in range(self.n)]
                cache[tup] = (FeDDHMultiClient.decrypt(cols, tag, pk, sk, self.bound) / s) / total
            out[i] = cache[tup]
        return out


def fe_secure_aggregate(dicts, num_samples, kappa=64, tag=b"efu"):
    """Cluster + encrypt + DMCFE-aggregate client weights into a weighted average.

    Returns a state dict of the aggregated weights (BatchNorm buffers averaged in
    plaintext and merged back).
    """
    n = len(dicts)
    dmcfe = DMCFE(n, total_samples=sum(num_samples))

    clustered = [cluster(sd, kappa) for sd in dicts]
    ciphertexts, mappings = [], []
    for cid, (Z, P, keys, shapes) in enumerate(clustered):
        ciphertexts.append(dmcfe.encrypt_centroids(cid, Z, tag))
        mappings.append(P)

    d = len(clustered[0][1]); keys, shapes = clustered[0][2], clustered[0][3]
    flat = dmcfe.secure_aggregate(ciphertexts, mappings, num_samples, tag, d)
    agg = _unflatten(flat, keys, shapes)

    # BatchNorm buffers are not weights: average them in plaintext and merge.
    total = sum(num_samples)
    for k, v in dicts[0].items():
        if _is_weight(k):
            continue
        if "num_batches_tracked" in k:
            agg[k] = v
        else:
            agg[k] = sum(sd[k].float() * (w / total) for sd, w in zip(dicts, num_samples))
    return agg
