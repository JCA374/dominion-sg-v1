"""Step 3: teach frozen c2 the teacher's verified improvements, starting from c2 itself.

    python supervise.py --labels sg_pilot --cohort <id> --anchors a1 --name s1 [--lam 1.0 --epochs 30]

Labels (bv-v1/data/<labels>, one cohort): a row is a verified improvement when the graded winner b (best paired value
among the candidates, strictly above the incumbent) also wins on the independent validation draws: gain VG > vmargin
and VG > vz x its SE. Loss (masked to the legal options, the policy at temperature 1 as PPO samples it):

    sum_i w_i softplus(-(logit_b - logit_inc)) / sum_i w_i       w_i = the verified gain (capped at 0.5)
  + lam  x KL(c2 || net) on an anchor batch (c2's own positions: actions, buys and subdecisions; anchors.py)
  + vlam x (v_net - v_c2)^2 on the same batch (the value head shares the trunk; rollout grades never touch it)

The label rows of the test split (held-out kingdoms) measure generalisation; held-out anchor games measure drift.
Out: models/<name>/net.pt + net.npy (rl-v3 format), results/supervise_<name>.json (per epoch: train and held-out pair
accuracy, anchor KL and argmax agreement by decision kind, value drift).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

from sg.paths import BV, C2, RESULTS, ROOT, use_bv, use_rl3

KIND_NAMES = {0: "action", 1: "buy"}


def verified(D, vmargin, vz):
    """(row index, winner, verified gain) of the verified improvements in a label dict."""
    q = np.where(D["CAND"].astype(bool), np.nan_to_num(D["Q"], nan=-1), -1)
    n = len(q)
    inc = D["INC"].astype(int)
    qi = q[np.arange(n), inc]
    b = q.argmax(1)
    better = q[np.arange(n), b] > qi
    ok = better & (D["VK"] > 0) & (D["VG"] > vmargin) & (D["VG"] > vz * D["VSE"])
    idx = np.where(ok)[0]
    return idx, b[idx], D["VG"][idx]


def load_labels(run, cohort):
    use_bv()
    sys.path.insert(0, BV)
    import importlib.util
    spec = importlib.util.spec_from_file_location("bv_train", os.path.join(BV, "train.py"))
    T = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(T)
    D = T.load(run, cohort=cohort)
    return D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--anchors", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--vlam", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--anchor_batch", type=int, default=2048)
    ap.add_argument("--pair_batch", type=int, default=256)
    ap.add_argument("--vmargin", type=float, default=0.02)
    ap.add_argument("--vz", type=float, default=2.0)
    ap.add_argument("--wcap", type=float, default=0.5)
    ap.add_argument("--kinds", default="buy,action", help="which verified labels teach: buy, action or both")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(4)
    torch.manual_seed(a.seed)
    use_rl3()
    from rl.net import from_flat, save
    D = load_labels(a.labels, a.cohort)
    kinds = {"buy": 1, "action": 0}
    keep_kind = np.isin(D["KIND"], [kinds[k] for k in a.kinds.split(",")])
    idx, win, gain = verified(D, a.vmargin, a.vz)
    sel = keep_kind[idx]
    idx, win, gain = idx[sel], win[sel], gain[sel]
    tr = D["SPLIT"][idx] == 2
    te = D["SPLIT"][idx] == 1
    A = dict(np.load(os.path.join(ROOT, "data", f"anchors_{a.anchors}.npz")))
    a_tr, a_te = np.where(A["SPLIT"] == 0)[0], np.where(A["SPLIT"] == 1)[0]
    c2 = from_flat(np.load(C2)).eval()
    net = from_flat(np.load(C2))
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    XL = torch.from_numpy(D["X"])
    LL = torch.from_numpy(D["LEGAL"]).bool()
    XA = torch.from_numpy(A["X"])
    LA = torch.from_numpy(A["LEGAL"]).bool()
    with torch.no_grad():                                   # c2's targets on every anchor state, once
        la, va = [], []
        for i in range(0, len(XA), 8192):
            lg, v = c2(XA[i:i + 8192], LA[i:i + 8192])
            la.append(torch.log_softmax(lg, -1))
            va.append(v)
        LPA, VA = torch.cat(la), torch.cat(va)
    W = torch.from_numpy(np.minimum(gain, a.wcap).astype(np.float32))
    B = torch.from_numpy(win.astype(np.int64))
    INC = torch.from_numpy(D["INC"][idx].astype(np.int64))
    I = torch.from_numpy(idx.astype(np.int64))

    def pair_stats(m):
        if not m.any():
            return None
        with torch.no_grad():
            lg, _ = net(XL[I[m]], LL[I[m]])
            d = lg.gather(1, B[m][:, None]).squeeze(1) - lg.gather(1, INC[m][:, None]).squeeze(1)
            top = lg.argmax(1)
        return {"n": int(m.sum()), "pair_acc": round(float((d > 0).float().mean()), 4),
                "takes_winner": round(float((top == B[m]).float().mean()), 4),
                "weighted_pair_acc": round(float(((d > 0).float() * W[m]).sum() / W[m].sum()), 4)}

    def anchor_stats(rows):
        out = {}
        with torch.no_grad():
            lg, v = net(XA[rows], LA[rows])
            lp = torch.log_softmax(lg, -1)
            p0 = LPA[rows].exp()
            kl = (p0 * (LPA[rows] - lp)).masked_fill(~LA[rows], 0).sum(-1)
            agree = lp.argmax(1) == LPA[rows].argmax(1)
            dv = (v - VA[rows]).abs()
        kind = A["KIND"][rows]
        for name, m in (("all", np.ones(len(rows), bool)), ("buy", kind == 1), ("action", kind == 0),
                        ("subdecisions", kind >= 2)):
            mt = torch.from_numpy(m)
            if m.any():
                out[name] = {"n": int(m.sum()), "kl": round(float(kl[mt].mean()), 5),
                             "argmax_agree": round(float(agree[mt].float().mean()), 4)}
        out["value_abs_drift"] = round(float(dv.mean()), 5)
        return out

    log = {"args": vars(a), "verified": {"train": int(tr.sum()), "test": int(te.sum()),
                                          "by_kind": {KIND_NAMES[k]: int((D["KIND"][idx] == k).sum()) for k in (0, 1)},
                                          "label_rows": int(len(D["K"])),
                                          "mean_gain": round(float(gain.mean()), 4) if len(gain) else None},
           "epochs": []}
    print(json.dumps(log["verified"]), flush=True)
    rng = np.random.default_rng(a.seed)
    trm = torch.from_numpy(tr)
    tem = torch.from_numpy(te)
    log["epochs"].append({"epoch": 0, "train": pair_stats(trm), "test": pair_stats(tem),
                          "anchor_test": anchor_stats(a_te)})
    t0 = time.time()
    tr_rows = np.where(tr)[0]
    for ep in range(1, a.epochs + 1):
        net.train()
        perm = rng.permutation(tr_rows)
        for s in range(0, len(perm), a.pair_batch):
            j = torch.from_numpy(perm[s:s + a.pair_batch])
            lg, _ = net(XL[I[j]], LL[I[j]])
            d = lg.gather(1, B[j][:, None]).squeeze(1) - lg.gather(1, INC[j][:, None]).squeeze(1)
            lpair = (W[j] * torch.nn.functional.softplus(-d)).sum() / W[j].sum()
            ar = torch.from_numpy(rng.choice(a_tr, a.anchor_batch))
            lga, v = net(XA[ar], LA[ar])
            lp = torch.log_softmax(lga, -1)
            kl = (LPA[ar].exp() * (LPA[ar] - lp)).masked_fill(~LA[ar], 0).sum(-1).mean()
            vl = ((v - VA[ar]) ** 2).mean()
            loss = lpair + a.lam * kl + a.vlam * vl
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        net.eval()
        e = {"epoch": ep, "train": pair_stats(trm), "test": pair_stats(tem), "anchor_test": anchor_stats(a_te),
             "secs": round(time.time() - t0)}
        log["epochs"].append(e)
        print(json.dumps({k: e[k] for k in ("epoch", "train", "test")}), "anchor KL",
              e["anchor_test"]["all"]["kl"], "agree", e["anchor_test"]["all"]["argmax_agree"], flush=True)
    out = os.path.join(ROOT, "models", a.name)
    os.makedirs(out, exist_ok=True)
    save(net, os.path.join(out, "net.pt"))
    log["net"] = os.path.relpath(os.path.join(out, "net.npy"), ROOT)
    json.dump(log, open(os.path.join(RESULTS, f"supervise_{a.name}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
