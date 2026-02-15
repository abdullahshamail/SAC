import numpy as np
def attr_align_from_indices(indices, user_to_key, keyToAttr, features, attr_names):
    emb_keys = [user_to_key[i+1] for i in indices]
    vecs = []
    for k in emb_keys:
        raw = features[k]
        nrm = np.linalg.norm(raw)
        v = raw / nrm if nrm > 0 else raw
        vecs.append(v)
    if not vecs:
        return {a: 0.0 for a in attr_names}

    V = np.vstack(vecs)                    
    S = V.sum(axis=0)                      
    Q = float((V * V).sum())               
    TOT = float(S @ S) - Q
    if V.shape[0] < 2 or TOT <= 1e-12:
        return {a: 0.0 for a in attr_names}


    out = {}
    d = V.shape[1]
    for a in attr_names:

        buckets = {}
        for row_idx, k in enumerate(emb_keys):
            val = keyToAttr[k][a]
            if val not in buckets:
                buckets[val] = []
            buckets[val].append(row_idx)

        WIN = 0.0
        for rows in buckets.values():
            if len(rows) == 0: 
                continue
            G = V[rows]                        
            Sg = G.sum(axis=0)
            Qg = float((G * G).sum())
            WIN += float(Sg @ Sg) - Qg


        out[a] = max(0.0, min(1.0, WIN / TOT))
    return out
