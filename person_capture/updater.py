#!/usr/bin/env python3
from __future__ import annotations
import os, sys, time, json, shutil, zipfile, subprocess, tempfile, threading, fnmatch
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

try:
    # PySide6 optional (we guard all UI calls)
    from PySide6 import QtCore, QtWidgets
except Exception:
    QtCore = None
    QtWidgets = None

GITHUB_REPO = "xmarre/person_capture"
GITHUB_HTTPS = f"https://github.com/{GITHUB_REPO}.git"
API_COMMITS = f"https://api.github.com/repos/{GITHUB_REPO}/commits"
API_REPO = f"https://api.github.com/repos/{GITHUB_REPO}"

CHECK_INTERVAL_SEC_DEFAULT = 4 * 3600  # every 4h
PIP_ENV_SKIP = "PERSON_CAPTURE_SKIP_PIP"

def _abs(p: str|Path) -> Path:
    return Path(p).resolve()

def _now() -> float:
    try:
        return time.time()
    except Exception:
        return 0.0

def _call(args: list[str], cwd: Optional[Path] = None, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(args, cwd=str(cwd) if cwd else None, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=check)

def _which(prog: str) -> Optional[str]:
    from shutil import which
    return which(prog)

def _read_url(url: str, timeout: int = 15) -> bytes:
    # stdlib to avoid extra deps, honor proxy envs
    import urllib.request
    headers = {"User-Agent": "PersonCapture-Updater/1.0"}
    # Optional: raise GitHub API rate limits if a token is present
    gh_tok = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if gh_tok:
        headers["Authorization"] = f"Bearer {gh_tok.strip()}"
        headers["Accept"] = "application/vnd.github+json"
        headers["X-GitHub-Api-Version"] = "2022-11-28"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()

def _repo_root_for(path: Path) -> Path:
    """
    Find install root robustly:
    - PERSON_CAPTURE_ROOT env wins (set PERSON_CAPTURE_ROOT=/where/the/app/lives)
    - frozen exe dir (PyInstaller)
    - script dir (argv[0])
    - module dir (path argument)
    Then ascend for .git or 'person_capture'.
    """
    root_env = os.environ.get("PERSON_CAPTURE_ROOT")
    if root_env:
        pe = Path(root_env).expanduser().resolve()
        if pe.exists():
            return pe
    bases: list[Path] = []
    try:
        if getattr(sys, "frozen", False):
            bases.append(Path(sys.executable).resolve().parent)
    except Exception:
        pass
    try:
        bases.append(Path(sys.argv[0]).resolve().parent)
    except Exception:
        pass
    bases.append(path)
    for base in bases:
        p = base
        for _ in range(9):
            if (p / ".git").exists() or (p / "person_capture").is_dir():
                return p
            if p.parent == p:
                break
            p = p.parent
    return path

def is_git_repo(repo: Path) -> bool:
    return (repo / ".git").exists() and bool(_which("git"))

def ensure_https_remote(repo: Path) -> None:
    try:
        if not is_git_repo(repo):
            return
        # Force origin to HTTPS for CI/user machines that accidentally cloned with SSH
        _call(["git", "remote", "set-url", "origin", GITHUB_HTTPS], cwd=repo)
    except Exception:
        pass

def get_git_version(repo: Path) -> str:
    try:
        if not is_git_repo(repo):
            return "(nogit)"
        out = _call(["git", "describe", "--tags", "--always"], cwd=repo).stdout.strip()
        return out or "(unknown)"
    except Exception:
        return "(unknown)"

def _git_tracking_remote(repo: Path) -> Optional[str]:
    try:
        # Prefer the current branch's upstream
        out = _call(["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"], cwd=repo).stdout.strip()
        return out or None
    except Exception:
        # Fallback to origin/HEAD -> origin/<default>
        try:
            out = _call(["git", "symbolic-ref", "--short", "refs/remotes/origin/HEAD"], cwd=repo).stdout.strip()
            return out or "origin/main"
        except Exception:
            return "origin/main"

def _git_need_updates(repo: Path) -> Tuple[bool, str, str]:
    """
    Returns (needs_update, local_sha, remote_sha)
    """
    if not is_git_repo(repo):
        return False, "", ""
    _call(["git", "fetch", "--tags", "--prune"], cwd=repo)
    local = _call(["git", "rev-parse", "HEAD"], cwd=repo).stdout.strip()
    upstream = _git_tracking_remote(repo) or "origin/main"
    remote = _call(["git", "rev-parse", upstream], cwd=repo).stdout.strip()
    if not local or not remote:
        return False, local, remote
    if local == remote:
        return False, local, remote
    # If ahead or behind we still offer update; user decides
    # Count commits remote has that local doesn't
    ahead = _call(["git", "rev-list", "--count", f"{local}..{remote}"], cwd=repo).stdout.strip()
    needs = True
    try:
        needs = int(ahead) > 0
    except Exception:
        needs = True
    return needs, local, remote

def _pip_install_if_needed(repo: Path, before_sha: str, after_sha: str, py: str) -> str:
    """
    Detect whether requirements changed between commits; if yes run pip install -r ...
    """
    try:
        if os.environ.get(PIP_ENV_SKIP, "").strip().lower() in ("1","true","yes","on"):
            return "pip skipped (env)"
        changed = _call(["git", "diff", "--name-only", f"{before_sha}..{after_sha}"], cwd=repo).stdout.splitlines()
        reqs = [n for n in changed if n.lower().startswith("requirements") and n.lower().endswith(".txt")]
        if not reqs:
            return "requirements unchanged"
        # Pick the most relevant requirements file (prefer windows / default)
        candidates = [Path(n) for n in reqs]
        target = "requirements.txt"
        for c in candidates:
            s = str(c).lower()
            if "windows" in s:
                target = str(c); break
        # Run pip
        txt = _call([
            py,
            "-m",
            "pip",
            "install",
            "-r",
            target,
            "--upgrade",
            "--upgrade-strategy",
            "only-if-needed",
        ], cwd=repo).stdout
        return f"pip ok: {target}\n{txt}"
    except Exception as e:
        return f"pip failed: {e}"

def git_update(repo: Path, autostash: bool = True) -> Tuple[bool, str]:
    """
    Returns (ok, message). On success, the repo is updated to upstream HEAD.
    """
    if not is_git_repo(repo):
        return False, "Not a git repo"
    ensure_https_remote(repo)
    try:
        prev = _call(["git", "rev-parse", "HEAD"], cwd=repo).stdout.strip()
        # prepare
        _call(["git", "fetch", "--tags", "--prune"], cwd=repo)
        # stash if dirty (so update doesn't fail)
        if autostash:
            try:
                status = _call(["git", "status", "--porcelain"], cwd=repo).stdout.strip()
                if status:
                    _call(["git", "stash", "push", "-u", "-m", "PersonCapture autostash"], cwd=repo)
            except Exception:
                pass
        # Rebase onto upstream
        upstream = _git_tracking_remote(repo) or "origin/main"
        pull = _call(["git", "pull", "--rebase", "--autostash"], cwd=repo)
        if pull.returncode != 0 and pull.stderr:
            # attempt a non-rebase pull
            _call(["git", "pull", upstream.split('/')[0], upstream.split('/')[1]], cwd=repo, check=True)
        # submodules (if ever added)
        try:
            _call(["git", "submodule", "update", "--init", "--recursive"], cwd=repo)
        except Exception:
            pass
        curr = _call(["git", "rev-parse", "HEAD"], cwd=repo).stdout.strip()
        if prev == curr:
            return True, "Already up to date."
        # deps?
        pip_msg = _pip_install_if_needed(repo, prev, curr, sys.executable)
        # post-update hook (optional)
        hook = repo / "scripts" / "post_update.py"
        if hook.exists():
            try:
                _call([sys.executable, str(hook)], cwd=repo)
            except Exception:
                pass
        return True, f"Updated to {curr[:7]}. {pip_msg}"
    except subprocess.CalledProcessError as e:
        return False, f"git failed: {e.stderr or e.stdout}"
    except Exception as e:
        return False, f"update error: {e}"

def _find_requirements_files(base: Path) -> list[Path]:
    # top-level only, conservative
    pats = ["requirements*.txt"]
    files: list[Path] = []
    try:
        for p in base.iterdir():
            if p.is_file():
                for pat in pats:
                    if fnmatch.fnmatch(p.name.lower(), pat):
                        files.append(p)
    except Exception:
        pass
    return sorted(files, key=lambda x: x.name)

def _select_req_file(files: list[Path]) -> Optional[Path]:
    if not files:
        return None
    # Prefer windows-specific if present, else the default
    for f in files:
        if "windows" in f.name.lower():
            return f
    for f in files:
        if f.name.lower() == "requirements.txt":
            return f
    return files[0]

def _download_zip_to(path: Path, url: str) -> Path:
    buf = _read_url(url, timeout=30)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(buf)
    return path

def _extract_zip(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)

def _latest_commit_sha(branch: str = "main") -> Optional[str]:
    try:
        data = json.loads(_read_url(f"{API_COMMITS}?per_page=1&sha={branch}").decode("utf-8"))
        if isinstance(data, list) and data:
            return data[0].get("sha")
    except Exception:
        pass
    return None

def _default_branch() -> str:
    try:
        data = json.loads(_read_url(API_REPO).decode("utf-8"))
        b = data.get("default_branch", "main")
        return b or "main"
    except Exception:
        return "main"

def stage_zip_update(repo: Path, branch: Optional[str] = None) -> Tuple[bool, str, Optional[str]]:
    """
    Download and stage the latest zipball of branch to repo/update_staged.
    Returns (ok, msg, staged_flag_file).
    """
    try:
        branch = branch or _default_branch()
        sha = _latest_commit_sha(branch)
        if not sha:
            return False, "Could not resolve latest commit", None
        url = f"https://github.com/{GITHUB_REPO}/zipball/{sha}"
        tmpd = Path(tempfile.mkdtemp(prefix="pc_update_"))
        zpath = tmpd / "update.zip"
        _download_zip_to(zpath, url)
        stage = repo / "update_staged"
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
        _extract_zip(zpath, stage)
        try:
            zpath.unlink(missing_ok=True)
            shutil.rmtree(tmpd, ignore_errors=True)
        except Exception:
            pass
        # The archive contains a top-level folder. Normalize: copy its contents up one level.
        entries = list(stage.iterdir())
        top = entries[0] if entries else None
        if top and top.is_dir():
            for item in top.iterdir():
                shutil.move(str(item), str(stage / item.name))
            try:
                shutil.rmtree(top, ignore_errors=True)
            except Exception:
                pass
        # Write a flag file for swap-on-next-start
        flag = repo / "update_pending.json"
        with open(flag, "w", encoding="utf-8") as f:
            json.dump({"sha": sha, "branch": branch, "staged_dir": str(stage)}, f)
        return True, f"Staged {sha[:7]} to {stage}", str(flag)
    except Exception as e:
        return False, f"stage error: {e}", None

def _pip_install_requirements(req_file: Path) -> str:
    try:
        if os.environ.get(PIP_ENV_SKIP, "").strip().lower() in ("1","true","yes","on"):
            return "pip skipped (env)"
        if not req_file.exists():
            return "requirements not present"
        proc = _call([sys.executable, "-m", "pip", "install", "-r", str(req_file),
                      "--upgrade", "--upgrade-strategy", "only-if-needed"], cwd=req_file.parent)
        if proc.returncode == 0:
            return f"pip ok: {req_file.name}\n{proc.stdout}"
        return f"pip failed (rc={proc.returncode}): {proc.stderr or proc.stdout}"
    except Exception as e:
        return f"pip failed: {e}"

def apply_staged_update(repo: Path) -> Tuple[bool, str]:
    """
    If update_pending.json exists, swap files from update_staged in place.
    Intended to be called very early at startup.
    """
    try:
        flag = repo / "update_pending.json"
        if not flag.exists():
            return False, "no pending update"
        with open(flag, "r", encoding="utf-8") as f:
            info = json.load(f)
        sha_applied = str(info.get("sha") or "")
        staged = Path(info.get("staged_dir") or (repo / "update_staged"))
        if not staged.exists():
            try:
                flag.unlink()
            except Exception:
                pass
            return False, "staged dir missing"
        # Check if requirements changed (best-effort, content compare)
        staged_reqs = _find_requirements_files(staged)
        staged_req = _select_req_file(staged_reqs)
        repo_reqs = _find_requirements_files(repo)
        repo_req = _select_req_file(repo_reqs)
        staged_bytes = staged_req.read_bytes() if staged_req and staged_req.exists() else b""
        repo_bytes = repo_req.read_bytes() if repo_req and repo_req.exists() else b""
        # Move all files/dirs from staged into repo (in-place overwrite)
        for item in staged.iterdir():
            dst = repo / item.name
            # Remove existing dst first to avoid merge weirdness
            if dst.exists():
                if dst.is_dir():
                    shutil.rmtree(dst, ignore_errors=True)
                else:
                    try:
                        dst.unlink()
                    except Exception:
                        pass
            shutil.move(str(item), str(dst))
        try:
            shutil.rmtree(staged, ignore_errors=True)
        except Exception:
            pass
        try:
            flag.unlink()
        except Exception:
            pass
        # Persist applied version for non-git installs so future checks don't re-prompt
        try:
            if sha_applied and not is_git_repo(repo):
                ver_tmp = repo / "VERSION.tmp"
                ver_dst = repo / "VERSION"
                ver_tmp.write_text(sha_applied + "\n", encoding="utf-8")
                os.replace(ver_tmp, ver_dst)  # atomic on POSIX/NTFS
        except Exception:
            pass
        # If requirements changed, install now (early in startup).
        pip_msg = ""
        try:
            new_req = _select_req_file(_find_requirements_files(repo))
            if staged_bytes and staged_bytes != repo_bytes and new_req:
                pip_msg = " | " + _pip_install_requirements(new_req)
        except Exception:
            pass
        return True, "swapped" + pip_msg
    except Exception as e:
        return False, f"swap error: {e}"

def _restart_self(extra_args: Optional[list[str]] = None) -> None:
    # Supports both "python script.py" and frozen exe bundles.
    if getattr(sys, "frozen", False):
        # In frozen apps, sys.executable IS the executable.
        args = [sys.executable, *(sys.argv[1:] or [])]
    else:
        py = sys.executable
        args = [py, sys.argv[0], *(sys.argv[1:] or [])]
    if extra_args:
        args.extend(extra_args)
    creationflags = 0
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    subprocess.Popen(args, close_fds=True, creationflags=creationflags)

# ------------------ Qt Integration ------------------

@dataclass
class UpdateResult:
    ok: bool
    message: str
    updated: bool
    method: str  # 'git' or 'zip' or 'none'
    new_sha: Optional[str] = None

class UpdateManager(QtCore.QObject if QtCore else object):
    """
    Signals (if Qt is available):
      progress(str), info(str), updateAvailable(str), upToDate(), updateFailed(str), updated(str)
    """
    if QtCore:
        progress = QtCore.Signal(str)
        info = QtCore.Signal(str)
        updateAvailable = QtCore.Signal(str)  # message
        upToDate = QtCore.Signal()
        updateFailed = QtCore.Signal(str)
        updated = QtCore.Signal(str)  # message

    def __init__(self, app_name: str = "PersonCapture GUI"):
        if QtCore:
            super().__init__()
        self.app_name = app_name
        # compute repo root
        here = _abs(Path(__file__)).parent
        self.repo = _repo_root_for(here)
        self._lock = threading.Lock()
        self._checking = False

    # ---- public API for GUI ----
    def maybe_apply_pending_at_start(self) -> None:
        ok, msg = apply_staged_update(self.repo)
        if ok and QtCore:
            try:
                self.info.emit(f"Applied staged update: {msg}")
            except Exception:
                pass

    def check_for_updates_async(self, branch: Optional[str], force: bool = False, throttle_sec: int = CHECK_INTERVAL_SEC_DEFAULT) -> None:
        if not QtCore:
            return
        with self._lock:
            if self._checking:
                return
            self._checking = True
        def _run():
            try:
                # Use the app's real org/app if available to avoid mismatched stores
                org = QtCore.QCoreApplication.organizationName() if QtCore else None
                if not org:
                    org = "PersonCapture"
                appn = QtCore.QCoreApplication.applicationName() if QtCore else None
                if not appn:
                    appn = self.app_name
                s = QtCore.QSettings(org, appn)
                last_t = float(s.value("update_last_check_t", 0.0))
                if (not force) and (_now() - last_t < float(throttle_sec)):
                    # silently ignore
                    return
                # path 1: git
                if is_git_repo(self.repo):
                    need, local, remote = _git_need_updates(self.repo)
                    s.setValue("update_last_check_t", _now()); s.sync()
                    if need:
                        msg = f"Update available: {local[:7]} → {remote[:7]}"
                        self.updateAvailable.emit(msg)
                    else:
                        self.upToDate.emit()
                    return
                # path 2: no git → compare commit sha via API to local version file (optional)
                remote_sha = _latest_commit_sha(branch or _default_branch())
                # Best-effort local version: look for a VERSION or git hash file dropped at build,
                # else assume we may be behind.
                ver = (self.repo / "VERSION")
                local_sha = ver.read_text(encoding="utf-8").strip()[:40] if ver.exists() else ""
                if remote_sha:
                    s.setValue("update_last_check_t", _now()); s.sync()
                    if local_sha and remote_sha.startswith(local_sha):
                        self.upToDate.emit()
                    else:
                        self.updateAvailable.emit(f"Update available (zip): {remote_sha[:7]}")
                else:
                    # Network/rate-limit hiccup — don't show a fake update, don't throttle retries
                    self.info.emit("Update check skipped (no remote SHA).")
            except Exception as e:
                self.updateFailed.emit(str(e))
            finally:
                with self._lock:
                    self._checking = False
        t = threading.Thread(target=_run, daemon=True)
        t.start()

    def perform_update_async(self, prefer_git: bool = True, branch: Optional[str] = None) -> None:
        if not QtCore:
            return
        def _run():
            try:
                if prefer_git and is_git_repo(self.repo):
                    self.progress.emit("Updating via git…")
                    ok, msg = git_update(self.repo, autostash=True)
                    if ok:
                        self.updated.emit(msg)
                        return
                    else:
                        # Fall back to zip if git failed (e.g., conflicts)
                        self.info.emit(f"git update failed, will try zip: {msg}")
                self.progress.emit("Staging zip update…")
                ok, msg, flag = stage_zip_update(self.repo, branch=branch or _default_branch())
                if ok:
                    self.updated.emit(msg)
                else:
                    self.updateFailed.emit(msg)
            except Exception as e:
                self.updateFailed.emit(str(e))
        t = threading.Thread(target=_run, daemon=True)
        t.start()

    def restart_now(self) -> None:
        _restart_self()
        if QtWidgets:
            QtWidgets.QApplication.quit()
        else:
            os._exit(0)

    # Convenience for About box / footer
    def version_string(self) -> str:
        if is_git_repo(self.repo):
            return get_git_version(self.repo)
        try:
            ver = (self.repo / "VERSION")
            if ver.exists():
                return ver.read_text(encoding="utf-8").strip() or "(unknown)"
        except Exception:
            pass
        return "(unknown)"
