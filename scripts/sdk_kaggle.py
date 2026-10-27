"""Minimal kagglesdk wrappers for ARC-AGI-2.

Credentials come from ~/.kaggle/kaggle.json. Kernels are pushed, polled and
submitted under that account (or KAGGLE_USERNAME if set); the placeholder
YOUR_KAGGLE_USERNAME in kernel-metadata.json is replaced by it at push time.

Usage:
  python scripts/sdk_kaggle.py comp                       # competition metadata (limits/deadline)
  python scripts/sdk_kaggle.py lb                         # leaderboard (top + cutoff)
  python scripts/sdk_kaggle.py push  <kernel_dir>         # push kernel from dir w/ kernel-metadata.json
  python scripts/sdk_kaggle.py status <kernel_slug>       # kernel session status
  python scripts/sdk_kaggle.py log    <kernel_slug>       # grep-able kernel run log
  python scripts/sdk_kaggle.py submit <kernel_slug> <version> "<description>"
  python scripts/sdk_kaggle.py subs                       # recent submissions + scores
"""
import json, os, sys
from pathlib import Path

from kagglesdk import KaggleClient
from kagglesdk.kernels.services.kernels_api_service import (
    ApiSaveKernelRequest, ApiGetKernelSessionStatusRequest,
    ApiListKernelSessionOutputRequest,
)
from kagglesdk.competitions.services.competition_api_service import (
    ApiCreateCodeSubmissionRequest, ApiListSubmissionsRequest,
    ApiGetCompetitionRequest, ApiGetLeaderboardRequest,
)

COMP = "arc-prize-2026-arc-agi-2"
SUBMISSION_FILE = "submission.json"
PLACEHOLDER = "YOUR_KAGGLE_USERNAME"


def _creds():
    return json.loads(Path(os.path.expanduser("~/.kaggle/kaggle.json")).read_text())


def _owner():
    return os.environ.get("KAGGLE_USERNAME") or _creds()["username"]


def _client():
    creds = _creds()
    c = KaggleClient()
    c.username = creds["username"]
    c.api_token = creds["key"]
    return c


def comp():
    c = _client()
    r = ApiGetCompetitionRequest()
    r.competition_name = COMP
    res = c.competitions.competition_api_client.get_competition(r)
    for f in ("max_daily_submissions", "deadline", "is_kernels_submissions_only",
              "max_team_size", "evaluation_metric", "submissions_disabled",
              "enabled_date", "reward", "category"):
        print(f"  {f}: {getattr(res, f, '?')}")


def lb():
    c = _client()
    r = ApiGetLeaderboardRequest()
    r.competition_name = COMP
    res = c.competitions.competition_api_client.get_leaderboard(r)
    rows = getattr(res, "submissions", None) or getattr(res, "entries", None) or []
    print(f"leaderboard rows: {len(rows)}")
    for i, s in enumerate(rows[:25], 1):
        team = getattr(s, "team_name", getattr(s, "teamName", "?"))
        score = getattr(s, "score", "?")
        print(f"  {i:>4}  {score:<10}  {team}")
    if len(rows) > 25:
        mid = rows[len(rows) // 2]
        print(f"  ... median(@{len(rows)//2}): {getattr(mid,'score','?')}  total teams: {len(rows)}")


def push(folder):
    c = _client()
    folder = Path(folder)
    meta = json.loads((folder / "kernel-metadata.json").read_text())
    code_path = folder / meta["code_file"]
    text = code_path.read_text()
    # Notebook: strip outputs + join source lists. Script: send raw text.
    if meta.get("kernel_type") == "notebook" or code_path.suffix == ".ipynb":
        nb = json.loads(text)
        for cell in nb.get("cells", []):
            if cell.get("cell_type") == "code":
                cell["outputs"] = []
            if isinstance(cell.get("source"), list):
                cell["source"] = "".join(cell["source"])
        text = json.dumps(nb)
    r = ApiSaveKernelRequest()
    r.id = None
    r.slug = meta["id"].replace(PLACEHOLDER, _owner())
    r.new_title = meta.get("title")
    r.text = text
    r.language = meta.get("language", "python")
    r.kernel_type = meta.get("kernel_type", "script")
    r.is_private = meta.get("is_private", True)
    r.enable_gpu = meta.get("enable_gpu", False)
    r.enable_tpu = meta.get("enable_tpu", False)
    r.enable_internet = meta.get("enable_internet", False)
    r.dataset_data_sources = meta.get("dataset_sources", [])
    r.competition_data_sources = meta.get("competition_sources", [])
    r.kernel_data_sources = [s.replace(PLACEHOLDER, _owner()) for s in meta.get("kernel_sources", [])]
    r.model_data_sources = meta.get("model_sources", [])
    r.category_ids = meta.get("keywords", [])
    r.docker_image = meta.get("docker_image", None)
    r.machine_shape = meta.get("machine_shape", None)
    res = c.kernels.kernels_api_client.save_kernel(r)
    print("PUSH OK:", getattr(res, "url", None), "| v", getattr(res, "version_number", "?"),
          "| err:", repr(getattr(res, "error", "")), "| bad_ds:", getattr(res, "invalid_dataset_sources", []))


def status(slug):
    c = _client()
    r = ApiGetKernelSessionStatusRequest()
    r.user_name = _owner()
    r.kernel_slug = slug
    res = c.kernels.kernels_api_client.get_kernel_session_status(r)
    print(slug, "->", res.status, "|", getattr(res, "failure_message", "") or "")


def log(slug):
    c = _client()
    r = ApiListKernelSessionOutputRequest()
    r.user_name = _owner()
    r.kernel_slug = slug
    r.page_size = 200
    res = c.kernels.kernels_api_client.list_kernel_session_output(r)
    raw = str(res.log)
    try:
        arr = json.loads(raw)
        text = "".join(d.get("data", "") for d in arr)
    except Exception:
        text = raw
    print(text)


def submit(slug, version, desc):
    c = _client()
    r = ApiCreateCodeSubmissionRequest()
    r.competition_name = COMP
    r.file_name = SUBMISSION_FILE
    r.kernel_owner = _owner()
    r.kernel_slug = slug
    r.kernel_version = int(version)
    r.submission_description = desc
    try:
        res = c.competitions.competition_api_client.create_code_submission(r)
        print("SUBMIT OK: ref=", res.ref, "| msg=", res.message)
    except Exception as e:
        resp = getattr(e, "response", None)
        if resp is not None:
            print(f"SUBMIT FAIL: HTTP {resp.status_code} {resp.text[:300]}")
        else:
            print("SUBMIT FAIL:", repr(e))


def subs():
    c = _client()
    r = ApiListSubmissionsRequest()
    r.competition_name = COMP
    res = c.competitions.competition_api_client.list_submissions(r)
    for s in res.submissions[:12]:
        print(f"  {s.ref}  {s.status}  pub={s.public_score}  priv={getattr(s,'private_score','')}  {(s.description or '')[:48]}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "comp"
    if cmd == "comp":
        comp()
    elif cmd == "lb":
        lb()
    elif cmd == "push":
        push(sys.argv[2])
    elif cmd == "status":
        status(sys.argv[2])
    elif cmd == "log":
        log(sys.argv[2])
    elif cmd == "submit":
        submit(sys.argv[2], sys.argv[3], sys.argv[4])
    elif cmd == "subs":
        subs()
    else:
        print("unknown cmd:", cmd)
