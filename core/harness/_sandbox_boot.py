"""
================================================================================
BLOCK: Sandbox Bootstrap (runs INSIDE the child interpreter)
--------------------------------------------------------------------------------
Role & contracts
  Executed as `python -I -S _sandbox_boot.py` by resilience.run_sandboxed_python.
  Reads untrusted code from stdin, installs an irreversible PEP-578 audit hook,
  then exec()s the code. Stdlib only; imports nothing from the repo.

Key engineering mechanism
  Audit hook denies: sockets/DNS, subprocess/exec/spawn/fork/pty, ctypes,
  imports of network/process/FFI modules, file writes outside the sandbox
  cwd, and file reads outside cwd + the stdlib. A blocked event writes a
  `SANDBOX_BLOCKED:<event>` marker to the raw stderr fd before raising, so
  the parent can attribute the block even if user code swallows the exception.

Failure modes & 1-step mitigation
  * Audit hooks are not a hard security boundary (PEP 578 says so). They are
    one layer: the parent also applies rlimits (CPU, address space, file size,
    fds), an empty env, `-I -S` isolation, a temp cwd and a wall-clock kill of
    the whole process group. For hostile multi-tenant code, run this inside
    gVisor/Firecracker - the interface stays identical.
================================================================================
"""
import os
import sys

_CODE = sys.stdin.read()
_CWD = os.path.realpath(os.getcwd())
_STDLIB = tuple({os.path.realpath(p) for p in (sys.prefix, sys.base_prefix, sys.exec_prefix, os.path.dirname(os.__file__))})
_ERR_FD = os.dup(2)
_BLOCKED_PREFIXES = (
    "socket.", "subprocess.", "os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.fork",
    "os.forkpty", "pty.", "ctypes.", "winreg.", "webbrowser.", "os.kill", "os.killpg", "signal.pthread_kill",
    "urllib.Request", "http.client.connect", "ftplib.", "smtplib.", "poplib.", "imaplib.", "telnetlib.",
    "os.chmod", "os.chown", "shutil.rmtree", "os.link", "os.symlink",
)
_BLOCKED_IMPORTS = {"socket", "_socket", "ssl", "_ssl", "subprocess", "_posixsubprocess", "multiprocessing",
                    "ctypes", "_ctypes", "pty", "asyncio", "selectors", "select", "mmap", "http", "urllib.request",
                    "ftplib", "smtplib", "webbrowser"}


def _deny(event: str):
    try:
        os.write(_ERR_FD, f"\nSANDBOX_BLOCKED:{event}\n".encode())
    except OSError:
        pass
    raise PermissionError(f"sandbox: '{event}' is not permitted")


def _inside(path: str, roots) -> bool:
    rp = os.path.realpath(path)
    return any(rp == r or rp.startswith(r + os.sep) for r in roots)


def _hook(event: str, args):
    if event.startswith(_BLOCKED_PREFIXES):
        _deny(event)
    if event == "import":
        mod = args[0] or ""
        if mod in _BLOCKED_IMPORTS or mod.split(".")[0] in {"socket", "ctypes", "subprocess", "multiprocessing"}:
            _deny(f"import:{mod}")
    elif event == "open":
        path, mode, flags = (list(args) + [None, None])[:3]
        if isinstance(path, int) or path is None:
            return
        path = os.fsdecode(path)
        writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
            isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND))
        if writing and not _inside(path, (_CWD,)):
            _deny(f"open-write:{path}")
        if not writing and not _inside(path, (_CWD,) + _STDLIB):
            _deny(f"open-read:{path}")
    elif event in ("os.listdir", "os.scandir") and args and args[0] not in (None, "."):
        if not _inside(os.fsdecode(args[0]), (_CWD,) + _STDLIB):
            _deny(f"{event}:{args[0]}")
    elif event in ("os.remove", "os.rename", "os.rmdir", "os.mkdir"):
        if args and isinstance(args[0], (str, bytes)) and not _inside(os.fsdecode(args[0]), (_CWD,)):
            _deny(f"{event}:{args[0]}")


sys.addaudithook(_hook)
del _hook
try:
    exec(compile(_CODE, "<sandbox>", "exec"), {"__name__": "__main__", "__builtins__": __builtins__})
except SystemExit:
    raise
except BaseException as e:  # report, non-zero exit
    sys.stderr.write(f"{type(e).__name__}: {e}\n")
    sys.exit(1)
