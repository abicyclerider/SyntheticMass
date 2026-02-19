"""RunPod pod lifecycle utilities shared by train/export/infer scripts."""

import os


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
