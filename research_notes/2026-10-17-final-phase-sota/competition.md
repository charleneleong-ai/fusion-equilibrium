# Competition channels, delta since 2026-10-11 (fetched 2026-10-17 ~11:45 UTC)

Primary sources: Codabench competition API (`/api/competitions/17739/`), leaderboard API
(`/api/phases/29580/get_leaderboard/`, pages 1-3, 146 rows), public Detailed Results pages for every leaderboard row
(`leaderboard/raw/details/<submission>.html`), and every issue / PR on
[Sophelio/fusion-equilibrium-challenge-starter](https://github.com/Sophelio/fusion-equilibrium-challenge-starter).

## Forum 17429: not readable without login

`https://www.codabench.org/forums/17429/` and the two thread URLs the organisers cite
([2807](https://www.codabench.org/forums/17429/2807/), the 23 Sep rule-3 announcement;
[2827](https://www.codabench.org/forums/17429/2827/), the rule-3 feedback and rule-6 compute answer) return
"Log In or Sign Up to view this competition forum". There is no public forum API (`/api/forums/...` 404). Per the
brief, no account was created and no login was attempted. **The forum was not read.** Its content is known only where
the organiser quoted it in starter-repo issues (#8, #9, #10, #12) and the Rules page; both are summarised below.
Reading the threads needs the user's own logged-in browser.

## Starter repo issues (8 issues, all closed; 4 PRs, all merged June-July; no Discussions)

| # | asker | date | what it settled |
|---|---|---|---|
| [5](https://github.com/Sophelio/fusion-equilibrium-challenge-starter/issues/5) | pshishkin | 08-12 | MAST core Thomson: `thomson_core_R[j]` pairs with `Te[:, j+1]`; channel 0 is all-NaN. Shipped as `align_mast_thomson_core` in `data_fixes.py` |
| [6](https://github.com/Sophelio/fusion-equilibrium-challenge-starter/issues/6) | pshishkin | 08-12 | DIII-D Ip time axis is a shared template; wrong on 6,322 / 9,113 shots. `fix_d3d_ip_times` |
| [7](https://github.com/Sophelio/fusion-equilibrium-challenge-starter/issues/7) | pshishkin | 08-15 | Ch2 gate moved to DIII-D composite S >= 0.85 (scorer v3.2.0 enforces) |
| [8](https://github.com/Sophelio/fusion-equilibrium-challenge-starter/issues/8) | shenmaa233 | 09-07 | Zero-shot = no target labels, not frozen params. TTA, MAST-input self-supervision, synthetic data calibrated on MAST *inputs* allowed |
| [9](https://github.com/Sophelio/fusion-equilibrium-challenge-starter/issues/9) | zhouyi-xiaoxiao, not-heavychevy | 09-08..10-02 | Per-machine pipelines allowed. **Choosing MAST variants by Development leaderboard score is permitted, disclose it.** Final compliance checked by code release + rebuild of top 5 + provenance review; an inaccurate provenance statement disqualifies by itself |
| [10](https://github.com/Sophelio/fusion-equilibrium-challenge-starter/issues/10) | namenguiin (team menguiin, MAST S 0.754) | 09-24 | CPU-only synthetic generation not capped (FreeGSNKE ~10 h on 64 cores); ship the set with SHA256 plus the code and seeds |
| [11](https://github.com/Sophelio/fusion-equilibrium-challenge-starter/issues/11) | anulum | 09-25..10-05 | **NEW.** Stuck submissions were Codabench 5xx errors (codalab/codabench#2557). The organisers' worker now retries. **In the Final, a submission lost to a platform fault is refunded on email** (now in the Rules) |
| [12](https://github.com/Sophelio/fusion-equilibrium-challenge-starter/issues/12) | Yiyi-philosophy | 10-07..10-11 | **NEW.** Thomson validity contract, below |

### Issue #12: Thomson validity contract (organiser answer, 2026-10-11)

- No length or mask field. A sample is valid iff its time is strictly greater than every earlier time in the same
  array (`t > maximum.accumulate(t)` shifted). Apply the same mask to Te and ne, **before** any `np.interp`
  or `searchsorted`.
- DIII-D Thomson: 0 means "no measurement" (no NaN, no negatives). About 47% of edge rows and 23% of core rows are
  all-zero. MAST uses NaN instead.
- `diii_d_train` has 10 edge arrays with zero-time padding after a reset. `diii_d_public_test` has none.
- **Final-phase inputs:** one DIII-D edge array has padding with a **constant non-zero time**, and one MAST core
  array has **one out-of-order sample followed by valid data**. Do not truncate at the first backward step, and do
  not drop t <= 0 (pre-breakdown samples go to -1.5 s).
- Dataset `Sophelio/fusion-equilibrium-challenge` revision `1e280905b85f2a6fdde7e06fca8cf3a1edf447cb` = v1.1.0, final;
  there will be no v1.1.1. Final inputs are a separate repo, **`Sophelio/fusion-equilibrium-challenge-final`,
  published 2026-10-19 00:00 UTC**, files named `diii_d_private_test.npz` / `mast_private_test.npz`.

Our [`profile_on_efit`](../../fusioneq/data.py#L74-L91) runs `searchsorted` on the raw time array with no
monotone mask, and [`thomson_features`](../../fusioneq/data.py#L138-L157) passes raw times. That is consistent with
how we trained, but it is undefined on the two Final arrays described above.

## Codabench pages (Rules / Evaluation / Data / Submission format), changes since 10-11

- Phases: Development ends 2026-10-18 00:00 UTC (11:00 AEDT). Final 2026-10-19 00:00 UTC to 2026-10-26 00:00 UTC
  (11:00 AEDT both ends), 3 submissions per team, leaderboard hidden.
- **Platform-fault refunds** in the Final (issue #11).
- **Plagiarism clause:** pointing `manifest.json` at another team's repo, or uploading their files, disqualifies
  the whole team. Relevant because two competitors now publish weights and code (AGPL-3.0 and MIT).
- The reproduction package must be on a **public** forge or host at submission time (Code URL field).
- The Data page still lists the **MAST solenoid as an open caveat**: `magnetics_sol_current` reads exactly 2x the
  FAIR-MAST level-2 `pf_active` value; the organisers have not said which is right. The geometry has 656 single-turn
  elements.
- Honorable mentions (best R²ψ, R²qb, Consistency, D_LCFS) are judged on the MAST private fold.
- The fact sheet has two fields only: `method_name` and `compute_light`, plus `code_url`.
- The Final leaderboard endpoint (`/api/phases/29581/get_leaderboard/`) is public and currently empty. If it lists
  submissions during the Final, their fact sheets (`code_url`) would be visible too.

## Final-phase logistics checklist (from the sources above)

1. Download `Sophelio/fusion-equilibrium-challenge-final` on 19 Oct 11:00 AEDT. Run inference on it with the Thomson
   monotone mask, or at least confirm that the two flagged arrays do not crash or misalign our features.
2. Code URL: public repo at a 40-char SHA, or a public archive with sha256. Weights included. Per-machine
   provenance. Declared GPU-h <= 24; CPU stages declared, not capped.
3. Disclose: Development scores used to select MAST variants; AI-assistant workflow (rule 4).
4. If a submission sticks or fails on the platform side, email `fusionec26@sophelio.io` to get the slot back.
5. Award #2 nomination email before 26 Oct 00:00 UTC if the Ch2 pick differs from the latest scored submission.
