import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

def attr_align_win_tot_from_vectors(normalized_vectors, attr_vals):
    """
    Attribute alignment as WIN/TOT fraction in [0,1].
    - normalized_vectors: (n,d) unit-norm rows
    - attr_vals: (n,) labels for ONE attribute (e.g., demographics per row)
    """
    V = normalized_vectors
    n = V.shape[0]
    if n < 2:
        return 0.0


    S = V.sum(axis=0)
    Q = float((V*V).sum())
    TOT = float(S @ S) - Q
    if TOT <= 1e-12:
        return 0.0

    buckets = defaultdict(list)
    for i, a in enumerate(attr_vals):
        buckets[a].append(i)

    WIN = 0.0
    for rows in buckets.values():
        if len(rows) == 0:
            continue
        G  = V[rows]
        Sg = G.sum(axis=0)
        Qg = float((G*G).sum())
        WIN += float(Sg @ Sg) - Qg

    align = WIN / TOT
    return float(max(0.0, min(1.0, align)))

def calculateDiversity_with_attr_win_tot(
    indices,
    userToEmb: pd.DataFrame,
    keyToVec: dict,
    keyToAttr: dict,
    attr_names=("demographics","sports","color")
):
    """
    Returns
    -------
    diversity : float  (avg pairwise cosine distance)
    r_c       : float  (||sum v|| / n)
    attr_align: dict   {attr_name: WIN/TOT fraction in [0,1]}
    n         : int
    """

    indices = np.array(list(indices)) + 1
    df = userToEmb.set_index('userId')
    emb_keys = df['embedding'].reindex(indices).to_numpy()

    vecs = [keyToVec[k] for k in emb_keys if k in keyToVec]
    n = len(vecs)
    if n == 0:
        return 0.0, 0.0, {a: 0.0 for a in attr_names}, 0

    V = np.vstack(vecs)
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    V = V / norms  


    sim = cosine_similarity(V)
    iu = np.triu_indices(n, 1)
    diversity = 1.0 - sim[iu].mean()

    S = V.sum(axis=0)
    r_c = float(np.linalg.norm(S) / n)

    attr_align = {}
    for a in attr_names:
        vals = [keyToAttr[int(k)][a] for k in emb_keys]  
        attr_align[a] = attr_align_win_tot_from_vectors(V, np.asarray(vals, dtype=object))

    return float(diversity), float(r_c), attr_align