# Resume

**Status 2026-11-24: finished.** All steps ran; nothing was promoted (results/decision.json). Units left running: sg-publish (the site).

Every step resumes where it stopped; run each as a systemd user unit (see CLAUDE.md), `<py>` = rl-v3's venv python.

| Step | Unit | Resume | Done when |
|---|---|---|---|
| 1 ablation | sg-ablation | `<py> -u ablation.py` (match.py `--resume`, frozen build/bv_ablation.so) | results/ablation.json has 4 arms |
| 2 pilot | sg-pipeline | `<py> -u pilot.py --act 0 --states 20000` (tops up labels; bank and survival skipped if present) | results/survive_sg_bank.json |
| 3 supervised | (manual) | `<py> supervise.py --labels sg_pilot --cohort a2c4633f925d --anchors a1 --name <n> --vz 2 --lam 100 --vlam 100 --epochs 100 --pair_batch 64`; then `bank_eval.py` and `netmatch.py --manifest dev` | chosen: models/s_z2_l100 |
| 4 PPO | sg-ppo | `<py> -u ppo.py --b models/s_z2_l100/net.npy --seeds s1,s2 --iterations 100 --workers 6` (each run `--continue`s) | rl-v3/runs/{A,B}_{s1,s2}/net_100.npy |
| 4 tests | sg-final | `<py> -u final.py --run pipeline --iteration 100` (matches `--resume`) | results/decision.json |

Site: unit sg-publish (`bash publish_loop.sh`, every 15 min). If a unit failed: `systemctl --user reset-failed`.
The promotion manifest is played once per candidate: never tune on it.
