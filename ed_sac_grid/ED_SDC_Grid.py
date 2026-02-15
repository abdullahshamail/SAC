from typing import Dict, List, Set
from .IDC import IDC
import numpy as np

from .utils import attr_align_from_indices, dbscan_labels_grid_onn

class ED_SDC_Grid:

    def __init__(self, eps, k, m, d_thresh, features, userToEmb, keyToAttr, attr_names):
        self.eps = float(eps)
        self.k = int(k)
        self.m = int(m)
        self.dth = float(d_thresh)

        self.features = features
        self.user_to_key: Dict[int, any] = dict(
            zip(userToEmb['userId'], userToEmb['embedding'])
        )

        self.N = 0
        self.t = -1
        self.active: Dict[int, IDC] = {}
        self.finished: List[IDC] = []
        self._next_id = 1

        self.keyToAttr = keyToAttr
        self.attr_names = attr_names

        self.E = None 


        self._ok_end: Dict[int, int] = {}
        self._ok_div: Dict[int, float] = {}
        self._ok_n:   Dict[int, int] = {}
        self._ok_idx: Dict[int, np.ndarray] = {}

        self._confirmed: Dict[int, bool] = {}


    def _build_E(self, N: int):
        if self.E is not None and self.E.shape[0] == N:
            return
        rows = []
        for i in range(N):
            key = self.user_to_key[i + 1] 
            v = np.asarray(self.features[key], dtype=float)
            n = np.linalg.norm(v)
            rows.append(v / n if n > 0 else v)
        self.E = np.vstack(rows)

    @staticmethod
    def _div(n: int, S: np.ndarray, Q: float) -> float:
        if n < 2:
            return 0.0
        return 1.0 - float((S @ S - Q) / (n * (n - 1)))

    def _start_from_idx(self, idx: np.ndarray, t: int) -> IDC:

        c = IDC(
            indices=set(idx.tolist()), is_assigned=True,
            start_time=t, end_time=t, diversity=0.0,
            embeddingKeys=None, embeddings=None, attr_names=self.attr_names,
        )
        V = self.E[idx]
        c.n = V.shape[0]
        c.S = V.sum(axis=0)
        c.Q = float(c.n) 
        c.diversity = self._div(c.n, c.S, c.Q)
        return c


    def _process_frame(self, P: np.ndarray):
        self.t += 1


        labels = dbscan_labels_grid_onn(P, self.eps, self.m)  

        label_order: List[int] = list(set(labels.tolist()))  
        label_pos: Dict[int, int] = {lab: pos for pos, lab in enumerate(label_order)}

        clusters: Dict[int, np.ndarray] = {}
        tmp_lists: Dict[int, List[int]] = {}
        for i, lab in enumerate(labels):
            lab = int(lab)
            lst = tmp_lists.get(lab)
            if lst is None:
                tmp_lists[lab] = [i]
            else:
                lst.append(i)
        for lab, lst in tmp_lists.items():
            if lst:
                clusters[lab] = np.fromiter(lst, dtype=np.int64)


        for conv in self.active.values():
            conv.is_assigned = False
        used: Set[int] = set()
        for cid, conv in list(self.active.items()):
            if not conv.indices:
                pass

            idx = np.fromiter(conv.indices, dtype=np.int64)
            labs_present = np.unique(labels[idx]).astype(int, copy=False)

            matched = False

            for lab in sorted((l for l in labs_present if l in label_pos),
                              key=lambda l: label_pos[l]):
                mask = (labels[idx] == lab)
                inter = idx[mask]
                if inter.size < self.m:
                    continue 
                if inter.size == idx.size:
                    new_n, new_S, new_Q = conv.n, conv.S, conv.Q
                else:
                    keep_mask = mask
                    leavers = idx[~keep_mask]
                    new_n = conv.n - leavers.size
                    new_S = conv.S - (self.E[leavers].sum(axis=0) if leavers.size else 0.0)
                    new_Q = conv.Q - float(leavers.size)

                if new_n <= 1:
                    continue
                new_div = self._div(new_n, new_S, new_Q)
                if new_div < self.dth:
                    continue

                if inter.size != idx.size:
                    conv.S, conv.Q, conv.n = new_S, new_Q, new_n
                    conv.indices = set(inter.tolist())
                conv.end_time = self.t
                conv.diversity = new_div
                conv.r_c = (np.linalg.norm(conv.S) / conv.n) if conv.n > 0 else 0.0
                conv.is_assigned = True

                self._ok_end[cid] = self.t
                self._ok_div[cid] = float(new_div)
                self._ok_n[cid]   = int(conv.n)
                self._ok_idx[cid] = inter.astype(np.int64, copy=True)

                life = (conv.end_time - conv.start_time) + 1
                if life >= self.k and not self._confirmed.get(cid, False):
                    conv.attr_align = attr_align_from_indices(
                            indices=conv.indices,
                            user_to_key=self.user_to_key,
                            keyToAttr=self.keyToAttr,
                            features=self.features,
                            attr_names=self.attr_names
                        )
                    self.finished.append(conv)  
                    self._confirmed[cid] = True

                used.add(int(lab))
                matched = True
                break

            if not matched:
                if not self._confirmed.get(cid, False):
                    ok_end = self._ok_end.get(cid, conv.end_time)
                    ok_div = self._ok_div.get(cid, conv.diversity)
                    ok_n   = self._ok_n.get(cid, conv.n)
                    ok_idx = self._ok_idx.get(
                        cid, np.fromiter(sorted(conv.indices), dtype=np.int64)
                    )
                    life_ok = (ok_end - conv.start_time) + 1
                    if life_ok >= self.k and ok_n >= self.m and ok_div >= self.dth:
                        conv.end_time  = ok_end
                        conv.diversity = ok_div
                        conv.n         = ok_n
                        conv.indices   = set(ok_idx.tolist())
                        conv.attr_align = attr_align_from_indices(
                            indices=conv.indices,
                            user_to_key=self.user_to_key,
                            keyToAttr=self.keyToAttr,
                            features=self.features,
                            attr_names=self.attr_names
                        )
                        self.finished.append(conv)

                del self.active[cid]
                self._ok_end.pop(cid, None); self._ok_div.pop(cid, None)
                self._ok_n.pop(cid, None);   self._ok_idx.pop(cid, None)
                self._confirmed.pop(cid, None)

        for lab in label_order:
            if lab not in clusters:
                continue
            if lab in used:
                continue
            members = clusters[lab]
            if members.size < self.m:
                continue

            cand = self._start_from_idx(members, t=self.t)
            if cand.n >= self.m and cand.diversity >= self.dth:
                nid = self._next_id
                self._next_id += 1
                self.active[nid] = cand

                self._ok_end[nid] = self.t
                self._ok_div[nid] = float(cand.diversity)
                self._ok_n[nid]   = int(cand.n)
                self._ok_idx[nid] = members.astype(np.int64, copy=True)
                self._confirmed[nid] = False

    def fit_predict(self, X, y=None, sample_weight=None):
        N = len(X); T = len(X[0])

        def frame(t: int) -> np.ndarray:
            return np.vstack([np.asarray(X[i][t], dtype=float).ravel() for i in range(N)])

        self._build_E(N)
        self.N = N
        self.t = -1
        self.active.clear(); self.finished.clear()
        self._ok_end.clear(); self._ok_div.clear()
        self._ok_n.clear();   self._ok_idx.clear()
        self._confirmed.clear()
        self._next_id = 1

        P0 = frame(0)
        for t in range(T):
            Pt = frame(t)
            if Pt.shape != P0.shape:
                raise ValueError("Dynamic N or position dimension not supported.")
            self._process_frame(Pt)

        for cid, conv in list(self.active.items()):
            if not self._confirmed.get(cid, False):
                ok_end = self._ok_end.get(cid, conv.end_time)
                ok_div = self._ok_div.get(cid, conv.diversity)
                ok_n   = self._ok_n.get(cid, conv.n)
                ok_idx = self._ok_idx.get(
                    cid, np.fromiter(sorted(conv.indices), dtype=np.int64)
                )
                life_ok = (ok_end - conv.start_time) + 1
                if life_ok >= self.k and ok_n >= self.m and ok_div >= self.dth:
                    conv.end_time  = ok_end
                    conv.diversity = ok_div
                    conv.n         = ok_n
                    conv.indices   = set(ok_idx.tolist())
                    conv.attr_align = attr_align_from_indices(
                            indices=conv.indices,
                            user_to_key=self.user_to_key,
                            keyToAttr=self.keyToAttr,
                            features=self.features,
                            attr_names=self.attr_names
                        )
                    self.finished.append(conv)
            self._ok_end.pop(cid, None); self._ok_div.pop(cid, None)
            self._ok_n.pop(cid, None);   self._ok_idx.pop(cid, None)
            self._confirmed.pop(cid, None)

        self.active.clear()
        return self.finished
