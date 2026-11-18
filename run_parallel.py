#!/usr/bin/env python3
"""
Run FARSITE_full_fire.ipynb as several headless copies at once, one group of fires per copy.

Each copy runs Section 9 (full runs) and Section 11 (restart experiments) for its own fires,
using FARSITE_NCORES cores. Landscapes and weather are cached per fire, so copies don't clash.
When all copies finish, a quick summary-only pass of the notebook (no FARSITE) rebuilds the
combined fires/summary.csv, fires/experiments_summary.csv and figure from every fire's results.

Usage (from the notebook's folder, with the gdal_env environment active):

    python run_parallel.py --max-days 1          # quick end-to-end check (~1 simulated day per fire)
    python run_parallel.py                       # the full run
    python run_parallel.py --jobs A C            # only some jobs
    python run_parallel.py --dry-run             # plans only (no FARSITE)

    # keep it going after you close the terminal / browser tab:
    nohup python run_parallel.py > runs/launcher.log 2>&1 &

Progress:   tail -f runs/run_A.log            (one live log per job)
Results:    fires/<fire>/...  and  runs/run_A.ipynb  (executed notebook with figures)
Stop:       Ctrl-C (or kill the launcher), then check `pgrep -fa TestFARSITE` for leftovers
"""
import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# ----------------------------------------------------------------------------- settings
NOTEBOOK = 'FARSITE_run_parallel.ipynb'

# Balanced by FARSITE steps (full run + both experiments). Bobcat and CZU have the biggest
# perimeters (slowest steps), so each gets its own copy.
JOBS = {
    'A': ['Bobcat_2020'],                 # ~1,960 steps
    'B': ['CZU_2020'],                    # ~1,920 steps
    'C': ['Glass_2020'],                  # ~1,810 steps
    'D': ['Zogg_2020', 'Airport_2024'],   # ~2,620 steps (Airport has no experiments)
}
NCORES_PER_JOB   = 4          # Minimum for a FARSITE run
SPOT_PROBABILITY = None       # None = the notebook's SPOT_PROBABILITY; e.g. 0 to switch spotting off
KERNEL           = None       # None = kernel saved in the notebook; else a name from `jupyter kernelspec list`
STATUS_EVERY_MIN = 10         # how often the launcher prints a status line per job
STAGGER_SEC      = 20         # gap between job starts (kernels starting together can time out)
RUNS_DIR         = Path('runs')
# -----------------------------------------------------------------------------


def last_line(path):
    try:
        lines = [l.strip() for l in Path(path).read_text(errors='replace').splitlines() if l.strip()]
        return lines[-1][:140] if lines else '(no output yet)'
    except FileNotFoundError:
        return '(no log yet)'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--jobs', nargs='+', choices=list(JOBS), default=list(JOBS), help='which jobs to run')
    ap.add_argument('--dry-run', action='store_true', help='plan only, no FARSITE')
    ap.add_argument('--max-days', type=float, help='cap every fire to this many simulated days (quick check)')
    ap.add_argument('--ncores', type=int, default=NCORES_PER_JOB, help='FARSITE threads per job')
    ap.add_argument('--spot', type=float, default=SPOT_PROBABILITY, help='FARSITE spot probability override')
    ap.add_argument('--kernel', default=KERNEL, help='Jupyter kernel name (default: the notebook\'s)')
    ap.add_argument('--skip-experiments', action='store_true', help='Section 9 only')
    args = ap.parse_args()

    if not Path(NOTEBOOK).exists():
        sys.exit(f'{NOTEBOOK} not found. Run this from the notebook\'s folder.')
    RUNS_DIR.mkdir(exist_ok=True)

    procs = {}
    for name in args.jobs:
        log = RUNS_DIR / f'run_{name}.log'
        log.write_text('')                                   # fresh live log
        env = dict(os.environ,
                   FARSITE_FIRES=','.join(JOBS[name]),
                   FARSITE_NCORES=str(args.ncores),
                   FARSITE_DRY_RUN='1' if args.dry_run else '0',
                   FARSITE_LOG=str(log.resolve()))
        if args.spot is not None:
            env['FARSITE_SPOT_PROBABILITY'] = str(args.spot)
        if args.max_days is not None:
            env['FARSITE_MAX_SIM_DAYS'] = str(args.max_days)
        if args.skip_experiments:
            env['FARSITE_SKIP_EXPERIMENTS'] = '1'
        cmd = [sys.executable, '-m', 'jupyter', 'nbconvert', '--to', 'notebook', '--execute', NOTEBOOK,
               '--output-dir', str(RUNS_DIR), '--output', f'run_{name}',
               '--ExecutePreprocessor.timeout=-1', '--ExecutePreprocessor.startup_timeout=300', '--allow-errors']
        if args.kernel:
            cmd.append(f'--ExecutePreprocessor.kernel_name={args.kernel}')
        if procs:
            time.sleep(STAGGER_SEC)                          # don't start all kernels at the same moment
        nbc_log = open(RUNS_DIR / f'run_{name}.nbconvert.log', 'w')
        procs[name] = (subprocess.Popen(cmd, env=env, stdout=nbc_log, stderr=subprocess.STDOUT,
                                        start_new_session=True), log, time.time())
        print(f'started job {name}: {", ".join(JOBS[name])}  (log: {log})', flush=True)

    def stop(*_):
        print('\nstopping all jobs...', flush=True)
        for p, _, _ in procs.values():
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        sys.exit(1)
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    t_last = 0
    while True:
        running = [n for n, (p, _, _) in procs.items() if p.poll() is None]
        if time.time() - t_last >= STATUS_EVERY_MIN * 60 or not running:
            t_last = time.time()
            print(f'\n[{time.strftime("%H:%M")}]', flush=True)
            for n, (p, log, t0) in procs.items():
                state = 'running' if p.poll() is None else f'finished (exit {p.returncode})'
                print(f'  {n} {state:20s} {(time.time()-t0)/3600:5.1f} h | {last_line(log)}', flush=True)
        if not running:
            break
        time.sleep(15)

    bad = [n for n, (p, _, _) in procs.items() if p.returncode != 0]
    print('\nAll jobs finished.' + (f' Problems in: {bad} (see runs/run_<job>.nbconvert.log)' if bad else ''))

    # final pass: rebuild the combined tables/figure from every fire's saved results (no FARSITE)
    if not args.dry_run:
        print('Building combined summaries (runs/summary.ipynb)...', flush=True)
        env = dict(os.environ, FARSITE_SUMMARY_ONLY='1', FARSITE_DRY_RUN='0')
        cmd = [sys.executable, '-m', 'jupyter', 'nbconvert', '--to', 'notebook', '--execute', NOTEBOOK,
               '--output-dir', str(RUNS_DIR), '--output', 'summary',
               '--ExecutePreprocessor.timeout=-1', '--ExecutePreprocessor.startup_timeout=300', '--allow-errors']
        if args.kernel:
            cmd.append(f'--ExecutePreprocessor.kernel_name={args.kernel}')
        with open(RUNS_DIR / 'summary.nbconvert.log', 'w') as fh:
            subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
    print('Combined results: fires/summary.csv, fires/experiments_summary.csv, fires/experiments_iou_vs_lead.png')
    print('Per-job notebooks with figures: runs/run_<job>.ipynb; combined: runs/summary.ipynb')


if __name__ == '__main__':
    main()
