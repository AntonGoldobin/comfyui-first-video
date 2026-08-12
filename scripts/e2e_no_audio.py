"""Diagnostic E2E: strip LTXV audio nodes from minimal workflow.

The full api-workflow-minimal.json uses LTXVAudioVAEDecode which ComfyUI
rejects with 'missing_node_type'. Strip audio nodes + their connections
to test the visual-only pipeline.
"""
import json, os, time, sys
import requests

KEY = os.environ["API_KEY"]
EID = os.environ["ENDPOINT_ID"]
POLL = int(os.environ.get("POLL_TIMEOUT_S", "900"))

wf = json.load(open("/Volumes/SSDNSKIY/VSCODE/comfyui-first-video/api-workflow-minimal.json"))

# Drop audio-side nodes and their dependents (anything only connected via audio)
AUDIO_NODES = {"LTXVAudioVAEDecode", "LTXVEmptyLatentAudio", "LTXVConcatAVLatent", "LTXVSeparateAVLatent"}
stripped = []
for nid, node in list(wf.items()):
    if not isinstance(node, dict):
        continue
    if node.get("class_type") in AUDIO_NODES:
        wf.pop(nid)
        stripped.append(nid)

# Re-point VHS_VideoCombine audio input to nothing (drop it)
for nid, node in wf.items():
    if isinstance(node, dict) and node.get("class_type") == "VHS_VideoCombine":
        node["inputs"].pop("audio", None)

print(f"stripped audio nodes: {stripped}")
print(f"remaining nodes: {len(wf)}")

url = f"https://api.runpod.ai/v2/{EID}/run"
r = requests.post(url, json={"input": {"workflow": wf}}, headers={"Authorization": f"Bearer {KEY}"}, timeout=30)
print("submit:", r.status_code, r.text[:200])
if r.status_code != 200:
    sys.exit(1)
jid = r.json()["id"]
print(f"job: {jid}")

# Poll
deadline = time.time() + POLL
while time.time() < deadline:
    sr = requests.get(f"https://api.runpod.ai/v2/{EID}/status/{jid}", headers={"Authorization": f"Bearer {KEY}"}, timeout=10)
    sd = sr.json()
    print(f"  t={int(time.time())%10000:5d}s  status={sd.get('status')}")
    if sd.get("status") in ("COMPLETED", "FAILED", "CANCELLED"):
        print(json.dumps(sd, indent=2)[:2000])
        sys.exit(0 if sd.get("status") == "COMPLETED" else 2)
    time.sleep(15)
print("timeout")
sys.exit(4)
