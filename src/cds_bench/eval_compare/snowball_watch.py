"""Refresh the local full report and audit completion, without chat schedules."""

import fcntl
import json
import subprocess
import sys
import time

from cds_bench.eval_luna.io import save
from cds_bench.eval_luna.run import now

from .snowball_run import RAW, ROOT


def run(module):
    subprocess.run([sys.executable,'-m',module],cwd=ROOT,check=True)


def main():
    with (RAW/'watcher.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        last_count, last_state, last_time = -1,None,0
        while True:
            job = json.loads((RAW/'job-status.json').read_text())
            terminal = job['state'] in {'complete','failed','stopped'}
            count = job.get('completed',0)
            try:
                due = (count>=last_count+25 or job['state']!=last_state or
                       (count>last_count and time.monotonic()-last_time>=60) or terminal)
                if due:
                    run('cds_bench.eval_compare.report')
                    last_count,last_state,last_time = count,job['state'],time.monotonic()
                if job['state']=='complete':
                    run('cds_bench.eval_compare.snowball_full')
                    run('cds_bench.eval_compare.report')
                save(RAW/'watch-status.json',{'state':'finished' if terminal else 'watching',
                     'completed':count,'updated_at':now(),'error':None,
                     'publication':'Local report only; remote PR publication requires reviewed artifacts.'})
                if terminal:
                    if json.loads((RAW/'job-status.json').read_text())['state']!=job['state']:
                        continue
                    return
            except Exception as exc:
                save(RAW/'watch-status.json',{'state':'attention','completed':count,
                     'updated_at':now(),'error':f'{type(exc).__name__}: {exc}'})
                print(f'Report/audit error: {type(exc).__name__}: {exc}',flush=True)
            time.sleep(20)


if __name__=='__main__':
    main()
