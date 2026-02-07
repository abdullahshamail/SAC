from numba.types import UniTuple, int32
from numba import njit, types
from numba.typed import Dict as NumbaDict, List as NumbaList
import numpy as np
KEY2D = types.UniTuple(types.int32, 2)
KEY3D = types.UniTuple(types.int32, 3)
I32   = types.int32



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

@njit(cache=True, fastmath=True)
def _union_pass_core_2d(P, ucells, members, starts, cell_map, offsets, eps2, core_mask, parent, rank):
    K = ucells.shape[0]
    for ka in range(K):
        a0 = starts[ka]; a1 = starts[ka+1]
        ax = ucells[ka, 0]; ay = ucells[ka, 1]
        for oi in range(offsets.shape[0]):
            nx = ax + offsets[oi, 0]; ny = ay + offsets[oi, 1]
            key = (np.int32(nx), np.int32(ny))
            if key in cell_map:
                kb = cell_map[key]
            else:
                continue
            b0 = starts[kb]; b1 = starts[kb+1]
            if kb == ka:
                for ia in range(a0, a1):
                    i = members[ia]
                    if not core_mask[i]: continue
                    pix = P[i, 0]; piy = P[i, 1]
                    for jb in range(ia + 1, a1):
                        j = members[jb]
                        if not core_mask[j]: continue
                        dx = pix - P[j, 0]; dy = piy - P[j, 1]
                        d2 = dx*dx + dy*dy
                        if d2 <= eps2:
                            _union(parent, rank, i, j)
            else:
                if kb < ka: 
                    continue
                for ia in range(a0, a1):
                    i = members[ia]
                    if not core_mask[i]: continue
                    pix = P[i, 0]; piy = P[i, 1]
                    for jb in range(b0, b1):
                        j = members[jb]
                        if not core_mask[j]: continue
                        dx = pix - P[j, 0]; dy = piy - P[j, 1]
                        d2 = dx*dx + dy*dy
                        if d2 <= eps2:
                            _union(parent, rank, i, j)

@njit(cache=True, fastmath=True)
def _border_attach_2d(P, ucells, members, starts, cell_map, offsets, eps2, core_mask, labels, inv_cell):
    N = P.shape[0]
    for u in range(N):
        if core_mask[u]: continue
        cu = inv_cell[u]
        cx = ucells[cu, 0]; cy = ucells[cu, 1]
        pux = P[u, 0]; puy = P[u, 1]
        best = -1
        for oi in range(offsets.shape[0]):
            nx = cx + offsets[oi, 0]; ny = cy + offsets[oi, 1]
            key = (np.int32(nx), np.int32(ny))
            if key in cell_map:
                kb = cell_map[key]
            else:
                continue
            b0 = starts[kb]; b1 = starts[kb+1]
            for jb in range(b0, b1):
                v = members[jb]
                if not core_mask[v]: continue
                dx = pux - P[v, 0]; dy = puy - P[v, 1]
                d2 = dx*dx + dy*dy
                if d2 <= eps2:
                    lab = labels[v]
                    if lab >= 0 and (best == -1 or lab < best):
                        best = lab
        if best >= 0:
            labels[u] = best


@njit(cache=True)
def _pack_cells_2d_njit(coords: np.ndarray):
    N = coords.shape[0]
    cell_map = NumbaDict.empty(key_type=KEY2D, value_type=I32)

    key_x = NumbaList.empty_list(I32)
    key_y = NumbaList.empty_list(I32)
    counts = NumbaList.empty_list(I32)

    inv = np.empty(N, dtype=np.int32)

    K = 0
    for i in range(N):
        x = np.int32(coords[i, 0]); y = np.int32(coords[i, 1])
        key = (x, y)
        if key in cell_map:
            cid = cell_map[key]
        else:
            cid = np.int32(K)
            cell_map[key] = cid
            key_x.append(x); key_y.append(y)
            counts.append(np.int32(0))
            K += 1
        inv[i] = cid
        ci = int(cid)                
        counts[ci] = counts[ci] + 1

    starts = np.empty(K + 1, dtype=np.int32)
    starts[0] = 0
    s = 0
    for k in range(K):
        s += counts[k]
        starts[k + 1] = s

    members = np.empty(N, dtype=np.int32)
    write_ptr = starts.copy()
    for i in range(N):
        cid = inv[i]
        pos = write_ptr[cid]
        members[pos] = i
        write_ptr[cid] = pos + 1

    ucells = np.empty((K, 2), dtype=np.int32)
    for k in range(K):
        ucells[k, 0] = key_x[k]
        ucells[k, 1] = key_y[k]

    return ucells, members, starts, inv


def _pack_cells(coords: np.ndarray):
    coords = np.asarray(coords, dtype=np.int32)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must be (N,2) int32")
    if coords.shape[1] == 2:
        return _pack_cells_2d_njit(coords)



@njit(cache=True, fastmath=True)
def _count_edges_2d(P, ucells, members, starts, cell_map, offsets, eps2):
    N = P.shape[0]
    deg = np.zeros(N, dtype=np.int32)
    E = 0
    K = ucells.shape[0]
    for ka in range(K):
        a0 = starts[ka]; a1 = starts[ka+1]
        ax = ucells[ka,0]; ay = ucells[ka,1]
        for oi in range(offsets.shape[0]):
            nx = ax + offsets[oi,0]; ny = ay + offsets[oi,1]
            key = (np.int32(nx), np.int32(ny))
            if key in cell_map:
                kb = cell_map[key]
            else:
                continue
            b0 = starts[kb]; b1 = starts[kb+1]
            if kb == ka:
                for ia in range(a0, a1):
                    i = members[ia]
                    pix = P[i,0]; piy = P[i,1]
                    for jb in range(ia+1, a1):
                        j = members[jb]
                        dx = pix - P[j,0]; dy = piy - P[j,1]
                        d2 = dx*dx + dy*dy
                        if d2 <= eps2:
                            deg[i] += 1; deg[j] += 1
                            E += 1
            else:
                if kb < ka:     
                    continue
                for ia in range(a0, a1):
                    i = members[ia]
                    pix = P[i,0]; piy = P[i,1]
                    for jb in range(b0, b1):
                        j = members[jb]
                        dx = pix - P[j,0]; dy = piy - P[j,1]
                        d2 = dx*dx + dy*dy
                        if d2 <= eps2:
                            deg[i] += 1; deg[j] += 1
                            E += 1
    return deg, E


def _build_cell_map_2d(ucells: np.ndarray):
    d = NumbaDict.empty(key_type=UniTuple(int32, 2), value_type=int32)
    K = ucells.shape[0]
    for k in range(K):
        key = (np.int32(ucells[k, 0]), np.int32(ucells[k, 1]))
        d[key] = np.int32(k)
    return d


def _neighbor_offsets_2d():
    return np.array([(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)], dtype=np.int32)

@njit(cache=True, fastmath=True)
def _find(parent, a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a

@njit(cache=True, fastmath=True)
def _union(parent, rank, a, b):
    ra = _find(parent, a)
    rb = _find(parent, b)
    if ra == rb:
        return
    if rank[ra] < rank[rb]:
        parent[ra] = rb
    elif rank[ra] > rank[rb]:
        parent[rb] = ra
    else:
        parent[rb] = ra
        rank[ra] = rank[ra] + 1


DENSE_DEGREE_CUTOVER = 128  

def dbscan_labels_grid_onn(P: np.ndarray, eps: float, m: int) -> np.ndarray:
    P = np.asarray(P, dtype=np.float64)
    N, D = P.shape
    if N == 0: return np.zeros(0, dtype=np.int32)
    if D != 2: raise ValueError("Only 2D.")
    eps = float(eps); eps2 = eps * eps

    coords = np.floor(P / eps).astype(np.int32)
    ucells, members, starts, inv = _pack_cells(coords)

    if D == 2:
        cell_map = _build_cell_map_2d(ucells)
        offsets  = _neighbor_offsets_2d()
        deg, E   = _count_edges_2d(P, ucells, members, starts, cell_map, offsets, eps2)

    core_mask = deg >= (m - 1)
    labels = np.full(N, -1, dtype=np.int32)
    if not np.any(core_mask):
        return labels

    parent = np.arange(N, dtype=np.int32)
    rank   = np.zeros(N, dtype=np.int8)

    if D == 2:
        _union_pass_core_2d(P, ucells, members, starts, cell_map, offsets, eps2, core_mask, parent, rank)

    roots = np.full(N, -1, dtype=np.int32)
    for i in range(N):
        if core_mask[i]:
            roots[i] = _find(parent, i)
    root_to_min = {}
    for i in range(N):
        r = roots[i]
        if r >= 0:
            if r in root_to_min:
                if i < root_to_min[r]:
                    root_to_min[r] = i
            else:
                root_to_min[r] = i
    sorted_roots = sorted(root_to_min.keys(), key=lambda r: root_to_min[r])
    root_to_label = {r: li for li, r in enumerate(sorted_roots)}

    for i in range(N):
        r = roots[i]
        if r >= 0:
            labels[i] = root_to_label[r]

    if D == 2:
        _border_attach_2d(P, ucells, members, starts, cell_map, offsets, eps2, core_mask, labels, inv)

    return labels