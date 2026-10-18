#!/usr/bin/env python3
"""Local, read-only dashboard. SSH is queried only when a viewer requests data."""
import argparse
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import statistics
import subprocess
import threading
import time
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'runs/cloud-preparation'

# Fixed read-only commands; never import the trainer, load checkpoints or scan audio.
REMOTE = r'''
import csv, json, math, subprocess, time
from pathlib import Path
root = Path(__RMFSR_ROOT__)
run = Path(__RMFSR_RUN__)
control = Path(__RMFSR_CONTROL__)
def read_json(path):
    try: return json.loads(path.read_text())
    except (OSError, ValueError): return {}
def tail(path, size):
    try:
        with path.open('rb') as f:
            f.seek(0, 2); end = f.tell(); start = max(0, end-size); f.seek(start)
            data = f.read()
        if start: data = data.partition(b'\n')[2]
        return data.decode(errors='replace')
    except OSError: return ''
metrics = run / 'metrics.jsonl'
rows = []
for line in tail(metrics, 700000).splitlines():
    try:
        row = json.loads(line)
        if isinstance(row.get('step'), int): rows.append(row)
    except (ValueError, AttributeError): pass
workers = {'trainer': [], 'controller': []}
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit(): continue
    try: args = (proc / 'cmdline').read_bytes().split(b'\0')
    except OSError: continue
    try: process_run = Path(args[args.index(b'--run') + 1].decode())
    except (ValueError, IndexError, UnicodeError): continue
    if not process_run.is_absolute(): process_run = root / process_run
    if process_run != run: continue
    for key, name in [('trainer', b'rmfsr.train'), ('controller', b'rmfsr.continuous_train')]:
        if name in args: workers[key].append(int(proc.name))
gpu = None
try:
    data = subprocess.check_output(['nvidia-smi', '--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu', '--format=csv,noheader,nounits'], text=True, timeout=5)
    fields = next(csv.reader(data.splitlines()))
    gpu = dict(name=fields[0].strip(), utilization=float(fields[1]), memory_used_mib=float(fields[2]), memory_total_mib=float(fields[3]), temperature_c=float(fields[4]))
except (OSError, ValueError, StopIteration, subprocess.SubprocessError): pass
checkpoint = run / 'latest.pt'
stat = checkpoint.stat() if checkpoint.exists() else None
result = dict(remote_time=time.time(), run_path=str(run), rows=rows[-500:], workers=workers, gpu=gpu,
    metrics_modified=metrics.stat().st_mtime if metrics.exists() else None,
    checkpoint=dict(modified=stat.st_mtime, bytes=stat.st_size) if stat else None,
    control=read_json(control / 'status.json'),
    config=read_json(run / 'training-config.json'),
    log=tail(run / 'console.log', 14000).splitlines()[-18:])
def clean(value):
    if isinstance(value, float) and not math.isfinite(value): return None
    if isinstance(value, dict): return {k: clean(v) for k,v in value.items()}
    if isinstance(value, list): return [clean(v) for v in value]
    return value
print(json.dumps(clean(result)))
'''


def remote_script(active):
    """Render Python literals, never shell substitutions, for the selected run."""
    from pathlib import PurePosixPath
    root = PurePosixPath(active.get('remote_root', '/workspace/rmfsr'))
    paths = {'__RMFSR_ROOT__': root,
             '__RMFSR_RUN__': PurePosixPath(active.get('training_run', str(root / 'runs/efficient'))),
             '__RMFSR_CONTROL__': PurePosixPath(active.get('continuous_control', str(root / 'runs/continuous')))}
    script = REMOTE
    for marker, path in paths.items():
        if not path.is_absolute() or '..' in path.parts or not path.is_relative_to(root):
            raise ValueError('Dashboard paths must stay inside the remote repository')
        script = script.replace(marker, repr(str(path)))
    return script


def summarize(raw, active):
    rows = raw['rows']
    step = rows[-1]['step'] if rows else 0
    target = raw.get('config', {}).get('steps', active.get('target_updates', 300000))
    samples = [r['seconds'] for r in rows[-100:]
               if r['step'] > 1 and isinstance(r.get('seconds'), (int, float)) and r['seconds'] > 0]
    seconds = statistics.mean(samples) if samples else None
    modified = raw.get('metrics_modified')
    age = max(0, raw['remote_time'] - modified) if modified else None
    phase = raw.get('control', {}).get('phase')
    if step >= target:
        status = 'complete'
    elif phase == 'failed':
        status = 'failed'
    elif not raw['workers']['trainer']:
        status = 'between_sessions' if raw['workers']['controller'] else 'inactive'
    elif age is None:
        status = 'starting'
    elif age > 120:
        status = 'waiting'
    else:
        status = 'training'
    return dict(raw, step=step, target=target, seconds_per_update=seconds,
                rate_sample_count=len(samples), update_age=age, status=status,
                estimated_remaining_seconds=(max(0, target-step)*seconds) if seconds else None,
                pod_id=active['pod_id'], region=active['region'], hourly_usd=active['gpu_hourly_usd'],
                fetched_at=time.time(), read_only=True)


class Reader:
    def __init__(self, state=STATE):
        self.state = state
        self.lock = threading.Lock()
        self.cached = None
        self.last_attempt = 0
        self.observations = deque(maxlen=120)

    def fetch(self):
        with self.lock:
            if self.cached and time.monotonic()-self.last_attempt < 8:
                return self.cached
            self.last_attempt = time.monotonic()
            try:
                active = json.loads((self.state / 'active-run.json').read_text())
                ep = active['ssh']
                hosts = self.state / 'ssh/known_hosts'
                args = ['/usr/bin/ssh', '-i', str(self.state/'ssh/rmfsr_ed25519'),
                        '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes',
                        '-o', 'StrictHostKeyChecking=yes', '-o', f'UserKnownHostsFile="{hosts}"',
                        '-o', 'ConnectTimeout=8', '-o', 'ServerAliveInterval=5',
                        '-o', 'ServerAliveCountMax=2', '-p', str(ep['port']),
                        ep['username']+'@'+ep['host'], "python - <<'DASHBOARD_READ_ONLY'\n"+remote_script(active)+"\nDASHBOARD_READ_ONLY"]
                result = subprocess.run(args, capture_output=True, text=True, timeout=25, check=True)
                data = summarize(json.loads(result.stdout), active)
                if self.observations and (self.observations[-1][2] != active['pod_id'] or data['step'] < self.observations[-1][1]):
                    self.observations.clear()
                self.observations.append((data['remote_time'], data['step'], active['pod_id']))
                start = self.observations[0]
                elapsed = data['remote_time']-start[0]
                data['observed_seconds_per_update'] = elapsed/(data['step']-start[1]) if elapsed >= 60 and data['step'] > start[1] else None
                data['observation_seconds'] = elapsed
                data['connection_error'] = None
                self.cached = data
            except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
                message = 'SSH connection timed out' if isinstance(exc, subprocess.TimeoutExpired) else 'Could not read the pod. Check SSH connectivity.'
                self.cached = dict(self.cached or {}, connection_error=message, attempted_at=time.time())
            return self.cached


def serve(port):
    reader = Reader()
    page = Path(__file__).with_suffix('.html').read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            authority = self.headers.get('Host', '')
            allowed = {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}
            origin = self.headers.get('Origin')
            if authority not in allowed or (origin and origin not in {'http://'+h for h in allowed}):
                self.send_error(403)
                return
            path = urlparse(self.path).path
            if path == '/':
                body, mime = page, 'text/html; charset=utf-8'
            elif path == '/api/status':
                body, mime = json.dumps(reader.fetch(), allow_nan=False).encode(), 'application/json'
            elif path == '/health':
                body, mime = b'{"service":"rmfsr-live-dashboard","read_only":true}', 'application/json'
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    location = {'url': f'http://127.0.0.1:{server.server_port}', 'read_only': True}
    (STATE/'live-dashboard.json').write_text(json.dumps(location, indent=2)+'\n')
    print(json.dumps(location), flush=True)
    server.serve_forever()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    serve(parser.parse_args().port)
