import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer

def pretty_print_convoys(convoys, title=None, max_indices=12, attr_order=("demographics","sports","color"), timeTaken = 0, n = 0, T=0, eps = 0.02, div_val = 0.2, k = 0):
    def get(obj, *names, default=None):
        for n in names:
            if hasattr(obj, n):
                v = getattr(obj, n)
                if v is not None:
                    return v
        if isinstance(obj, dict):
            for n in names:
                if n in obj and obj[n] is not None:
                    return obj[n]
        return default

    rows = []
    for cid, c in enumerate(convoys):
        indices = get(c, "indices") or []
        indices = list(indices) if not isinstance(indices, list) else indices
        indices_sorted = sorted(indices)

        start = get(c, "start_time", "start")
        end   = get(c, "end_time", "end")
        dur   = (end - start + 1) if (start is not None and end is not None) else None

        attrs = get(c, "attr_align", "sattribute", "attributes", default={}) or {}

        if not attr_order:
            attr_keys = sorted(attrs.keys())
        else:

            extras = [k for k in sorted(attrs.keys()) if k not in attr_order]
            attr_keys = list(attr_order) + extras

        rows.append({
            "cid": cid,
            "start": start,
            "end": end,
            "dur": dur,
            "size": len(indices_sorted),
            "assigned": bool(get(c, "is_assigned", "assigned", default=False)),
            "diversity": get(c, "diversity"),
            "cohesion": get(c, "cohesion", "r_c"),
            "attrs": {k: attrs.get(k) for k in attr_keys if k in attrs},
            "indices": indices_sorted
        })

    rows.sort(key=lambda r: (float('inf') if r["start"] is None else r["start"],
                             float('inf') if r["end"]   is None else r["end"]))


    if title:
        print(f"\n=== {title}, T = {T}, n = {n}, eps={eps}, d_div={div_val}, k = {k}, numConvoys = {len(convoys)}, runtime = {timeTaken} ===")
    header_left = "cid | start-end (dur) | size | assigned | diversity | cohesion"

    attr_hdr = " | ".join([f"attr:{k}" for k in (rows[0]["attrs"].keys() if rows else [])])
    header = header_left + ((" | " + attr_hdr) if attr_hdr else "") + " | indices"
    print(header)
    print("-" * len(header))

    if len(rows) > 5:
        rows = rows[:5]
    for r in rows:
        div = f"{r['diversity']:.4f}" if isinstance(r["diversity"], (int, float)) else str(r["diversity"])
        coh = f"{r['cohesion']:.4f}" if isinstance(r["cohesion"], (int, float)) else str(r["cohesion"])
        idxs = r["indices"]
        if len(idxs) > max_indices:
            idx_str = f"[{', '.join(map(str, idxs[:max_indices]))}, … +{len(idxs)-max_indices}]"
        else:
            idx_str = f"[{', '.join(map(str, idxs))}]"
        attr_str = " | ".join(
            f"{(f'{v:.4f}' if isinstance(v, (int,float)) else v)}"
            for v in r["attrs"].values()
        )
        left = f"{r['cid']:>3} | {r['start']}-{r['end']} ({r['dur']}) | {r['size']:>4} | {str(r['assigned']):^8} | {div:>8} | {coh:>8}"
        print(left + ((" | " + attr_str) if attr_str else "") + f" | {idx_str}")

    return len(convoys), timeTaken



def _norm_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / n

def neighbor_counts(P: np.ndarray, eps: float) -> np.ndarray:
    d2 = np.sum((P[:, None, :] - P[None, :, :])**2, axis=2)
    within = (d2 <= eps * eps)
    return within.sum(axis=1) - 1


@dataclass
class SemanticsConfig:
    color: List[str]
    sports: List[str]
    demographics: List[str]

DEFAULT_SEM = SemanticsConfig(
    color=['red','blue','green','yellow','purple','orange','brown','pink','gray','olive','cyan','magenta'],
    sports=["Football","Baseball","Basketball", "Cricket", "Hockey", "Soccer", "Tennis", "Golf", "Swimming", "Boxing"],
    demographics=["USA","Canada","Mexico","UK","Pakistan","Brazil","Bangladesh","China","India","Russia",
                   "Japan","Germany","France","Italy","Spain","Australia","Indonesia","Netherlands","Turkey",
                   "Switzerland","Sweden","Poland","Belgium","Norway","Austria","UAE"]
)

def _build_atomic_vectors_random(sem_cfg: SemanticsConfig, dim_per_attr: int, seed: int):
    rng = np.random.default_rng(seed)
    out = {}
    def make(label_list, prefix):
        for lab in label_list:
            key = f"{prefix}:{lab}"
            vec = rng.normal(0, 1, size=(dim_per_attr,))
            vec = vec / (np.linalg.norm(vec) + 1e-12)
            out[key] = vec
    make(sem_cfg.demographics, "demographics")
    make(sem_cfg.sports, "sports")
    make(sem_cfg.color, "color")
    return out

def _build_atomic_vectors_sbert(sem_cfg: SemanticsConfig, dim_per_attr: int = 384, model_name: str = "all-mpnet-base-v2"):
    model = SentenceTransformer(model_name)

    def enc(prefix, items):
        texts = [f"{prefix}:{x}" for x in items]
        embs  = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return {t:e for t, e in zip(texts, embs)}

    out = {}
    out.update(enc("demographics", sem_cfg.demographics))
    out.update(enc("sports", sem_cfg.sports))
    out.update(enc("color", sem_cfg.color))
    return out

def _combine_semantic_triplet(demographics: str, sports: str, color: str,
                              atomic_dict: Dict[str, np.ndarray]) -> Tuple[str, np.ndarray]:
    k1 = f"demographics:{demographics}"
    k2 = f"sports:{sports}"
    k3 = f"color:{color}"
    v1 = atomic_dict[k1]; v2 = atomic_dict[k2]; v3 = atomic_dict[k3]
    combined_key_str = f"{k1}; {k2}; {k3}"
    combined_vec = np.concatenate([v1, v2, v3], axis=0)
    return combined_key_str, combined_vec

def make_synthetic_like_real(

    N: int = 8000,
    T: int = 3000,
    D: int = 2,
    box_min: Tuple[float, float] = (40.5654, -74.2602),
    box_max: Tuple[float, float] = (40.9824, -73.6945),

    eps: float = 0.02,
    m: int = 10,
    k: int = 20,


    num_convoys: int = 60,
    convoy_size_range: Tuple[int, int] = (10, 20), 
    life_range: Tuple[int, int] = (20, 400),        


    center_step: float = 0.0010,
    jitter_frac: float = 0.30,
    background_frac: float = 0.30,
    background_drift: float = 0.003,


    semantics: SemanticsConfig = DEFAULT_SEM,
    semantics_mode: str = "sbert",   
    dim_per_attr: int = 384,           
    semantic_purity: float = 0.6,      
    purity_on: Tuple[bool,bool,bool] = (True, True, True),  

    seed: int = 7
):
    # assert D == 2
    rng = np.random.default_rng(seed)


    sizes = rng.integers(convoy_size_range[0], convoy_size_range[1] + 1, size=num_convoys)
    sizes = np.maximum(sizes, m)
    members = []
    used = 0
    for s in sizes:
        if used + s > N:
            break
        grp = np.arange(used, used + s, dtype=np.int64)
        members.append(grp)
        used += s
    num_convoys = len(members)
    background_ids = np.arange(used, N, dtype=np.int64)

    lifetimes = []
    Lmin, Lmax = life_range
    Lmax = min(Lmax, T)
    for _ in range(num_convoys):
        life = rng.integers(max(k, Lmin), Lmax + 1)
        start = rng.integers(0, T - life + 1)
        end = start + life - 1
        lifetimes.append((start, end))


    def rand_point():
        return rng.uniform(box_min, box_max, size=(D,))
    centers = []
    for _ in range(num_convoys):
        c = np.empty((T, D), dtype=float)
        c[0] = rand_point()
        for t in range(1, T):
            c[t] = c[t-1] + rng.normal(0.0, center_step, size=D)
        c = np.minimum(np.maximum(c, box_min), box_max)
        centers.append(c)


    if semantics_mode == "sbert":
        atomic = _build_atomic_vectors_sbert(semantics, dim_per_attr=dim_per_attr)
    else:
        atomic = _build_atomic_vectors_random(semantics, dim_per_attr=dim_per_attr, seed=seed)

    user_sem = {
        "demographics": np.empty(N, dtype=object),
        "sports":      np.empty(N, dtype=object),
        "color":       np.empty(N, dtype=object),
    }


    def sample_profile():
        return (
            rng.choice(semantics.demographics),
            rng.choice(semantics.sports),
            rng.choice(semantics.color),
        )

    for grp in members:
        base_nat, base_gen, base_col = sample_profile()
        n_pure = int(np.ceil(len(grp) * semantic_purity))
        pure_ids = rng.choice(grp, size=n_pure, replace=False)
        for i in pure_ids:
            user_sem["demographics"][i] = base_nat if purity_on[0] else rng.choice(semantics.demographics)
            user_sem["sports"][i]      = base_gen if purity_on[1] else rng.choice(semantics.sports)
            user_sem["color"][i]       = base_col if purity_on[2] else rng.choice(semantics.color)

        rest = np.setdiff1d(grp, pure_ids, assume_unique=True)
        for i in rest:
            user_sem["demographics"][i] = rng.choice(semantics.demographics)
            user_sem["sports"][i]      = rng.choice(semantics.sports)
            user_sem["color"][i]       = rng.choice(semantics.color)

    for i in background_ids:
        user_sem["demographics"][i] = rng.choice(semantics.demographics)
        user_sem["sports"][i]      = rng.choice(semantics.sports)
        user_sem["color"][i]       = rng.choice(semantics.color)

    keySpace: Dict[str, int] = {}
    keyVectorSpace: Dict[int, np.ndarray] = {}
    keyAttrSpace: Dict[int, Dict[str, object]] = {} 
    next_key = 0

    user_ids = np.arange(1, N+1, dtype=np.int64)
    emb_keys = np.empty(N, dtype=np.int64)

    for idx in range(N):
        nat = user_sem["demographics"][idx]
        gen = user_sem["sports"][idx]
        col = user_sem["color"][idx]
        key_str, vec = _combine_semantic_triplet(nat, gen, col, atomic)

        if key_str not in keySpace:
            keySpace[key_str] = next_key
            keyVectorSpace[next_key] = vec
            keyAttrSpace[next_key] = {"demographics": nat, "sports": gen, "color": col}
            next_key += 1

        emb_keys[idx] = keySpace[key_str]

    userToEmb = pd.DataFrame({"userId": user_ids, "embedding": emb_keys})
    features: Dict[int, np.ndarray] = keyVectorSpace 

    X = rng.uniform(box_min, box_max, size=(N, T, D))
    sigma = eps * jitter_frac

    for grp, (s, e), c in zip(members, lifetimes, centers):
        J = rng.normal(0.0, sigma, size=(grp.size, e - s + 1, D))
        X[grp[:, None], np.arange(s, e + 1), :] = c[s:e+1][None, :, :] + J

    B = background_ids
    nb = int(len(B) * background_frac)
    if nb > 0:
        walkers = rng.choice(B, size=nb, replace=False)
        for w in walkers:
            drift = rng.normal(0.0, background_drift, size=(T, D))
            drift = np.cumsum(drift, axis=0)
            X[w] = np.minimum(np.maximum(X[w] + drift, box_min), box_max)

    X_list: List[List[List[float]]] = [[X[i, t].tolist() for t in range(T)] for i in range(N)]

    meta = {
        'num_convoys': num_convoys,
        'members': members,
        'lifetimes': lifetimes,
        'center_step': center_step,
        'sigma': sigma,
        'box_min': box_min,
        'box_max': box_max,
        'semantics_mode': semantics_mode,
        'dim_per_attr': dim_per_attr,
        'semantic_purity': semantic_purity,
        'purity_on': purity_on,
        'unique_combined_keys': next_key,
    }
    return X_list, userToEmb, features, meta, keyAttrSpace


def calibrate_density(
    N=2000, T=50, eps=0.02, target_median=36.0, target_p95=290.0, seed=123,
    D=2, tries=12
):
    rng = np.random.default_rng(seed)
    best = None
    box_min = (40.5654, -74.2602)
    box_max = (40.9824, -73.6945)

    for _ in range(tries):
        jitter_frac = rng.uniform(0.20, 0.35)
        center_step = rng.uniform(0.0006, 0.0016)
        X, U, F, meta, attr = make_synthetic_like_real(
            N=N, T=T, D=D,
            eps=eps, m=10, k=20,
            num_convoys=20,
            convoy_size_range=(12, 25),
            life_range=(20, 40),
            center_step=center_step,
            jitter_frac=jitter_frac,
            background_frac=0.35,
            background_drift=0.003,
            seed=rng.integers(1e9)
        )

        P = np.vstack([np.asarray(X[i][0], dtype=float) for i in range(N)])
        nc = neighbor_counts(P, eps)
        med = float(np.median(nc)); p95 = float(np.percentile(nc, 95))
        score = abs(med - target_median) + 0.01 * abs(p95 - target_p95)
        cand = (score, jitter_frac, center_step, med, p95)
        if (best is None) or (cand < best):
            best = cand

    score, jf, cs, med, p95 = best
    return dict(jitter_frac=jf, center_step=cs, median=med, p95=p95, score=score)

