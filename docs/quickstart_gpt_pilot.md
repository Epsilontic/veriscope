# GPT Pilot Quickstart

This is the fastest way to run the current GPT-first MVP path using the nanoGPT runner and pilot harness. It shows a real control/injected workflow, contract-valid capsules, and the report/diff/calibration flow in one short path.

## 1. Prerequisites

Clone `nanoGPT`, prepare the Shakespeare dataset, and install Veriscope with the torch extra:

```bash
git clone https://github.com/karpathy/nanoGPT.git

cd nanoGPT
python data/shakespeare_char/prepare.py
cd ..

python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e ".[torch]"
```

## 2. Control run

```bash
bash scripts/pilot/run.sh ./out/pilot_control -- --dataset shakespeare_char --nanogpt_dir ./nanoGPT
```

This produces a healthy baseline capsule for the GPT-first MVP path.

## 3. Injected run

```bash
bash scripts/pilot/run.sh ./out/pilot_injected -- --dataset shakespeare_char --nanogpt_dir ./nanoGPT \
  --data_corrupt_at 2500 --data_corrupt_len 400 --data_corrupt_frac 0.15 --data_corrupt_mode permute
```

This intentionally corrupts the run to test gate behavior against the control.

## 4. Inspect artifacts

```bash
veriscope validate ./out/pilot_control
veriscope validate ./out/pilot_injected
veriscope report ./out/pilot_control --format text
veriscope report ./out/pilot_injected --format text
veriscope diff ./out/pilot_control ./out/pilot_injected
```

Expect two valid capsules, two interpretable text reports, and a contract-aware diff. The control run should read like a healthy baseline; the injected run should show different gate behavior or summary outcomes under the same workflow.

## 5. Generate calibration report

```bash
python scripts/pilot/score.py \
  --control-dir ./out/pilot_control \
  --injected-dir ./out/pilot_injected \
  --out calibration.json \
  --out-md calibration.md
```

This produces FAR, delay, and overhead metrics for the fixed window signature used by the two runs.

## 6. Optional: override a decision

```bash
veriscope override ./out/pilot_injected --status pass --reason "Known infrastructure noise"
```

Overrides append a manual judgement and governance entry without modifying the raw run artifacts.

## 7. What this quickstart demonstrates

- GPT-first MVP path
- contract-valid capsules
- control/injected calibration workflow

## 8. What it does not demonstrate

- universal integration across arbitrary loops
- distributed training support
- cross-window calibration portability
