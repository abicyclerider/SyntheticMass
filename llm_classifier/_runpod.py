"""RunPod pod lifecycle utilities shared by train/export/infer scripts."""

import os
import sys

LOG_PATH = "/tmp/runpod_output.log"


def setup_logging():
    """Tee stdout/stderr to a log file for crash debugging. No-op if file can't be opened."""
    try:
        log_file = open(LOG_PATH, "w")
    except OSError:
        return

    class _Tee:
        def __init__(self, original, log):
            self._original = original
            self._log = log

        def write(self, data):
            self._original.write(data)
            self._original.flush()
            try:
                self._log.write(data)
                self._log.flush()
            except OSError:
                pass

        def flush(self):
            self._original.flush()
            try:
                self._log.flush()
            except OSError:
                pass

        def __getattr__(self, name):
            return getattr(self._original, name)

    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)


def upload_log(repo_id):
    """Upload the captured log file to HF Hub. No-op if no log exists."""
    if not os.path.exists(LOG_PATH):
        return
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        api.upload_file(
            path_or_fileobj=LOG_PATH,
            path_in_repo="run.log",
            repo_id=repo_id,
        )
        print(f"  Log uploaded to {repo_id}/run.log")
    except Exception as e:
        print(f"  Warning: failed to upload log: {e}")


def stop_runpod_pod():
    """Stop the current RunPod pod via API. No-op when not on RunPod."""
    pod_id = os.environ.get("RUNPOD_POD_ID")
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not pod_id or not api_key:
        return
    try:
        import requests

        print(f"\nStopping RunPod pod {pod_id}...")
        resp = requests.post(
            "https://api.runpod.io/graphql",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "query": f'mutation {{ podStop(input: {{podId: "{pod_id}"}}) {{ id }} }}'
            },
            timeout=30,
        )
        resp.raise_for_status()
        print("  Pod stop requested.")
    except Exception as e:
        print(f"  Warning: failed to stop pod: {e}")
