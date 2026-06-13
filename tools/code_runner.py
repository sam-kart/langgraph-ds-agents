import os
import shutil
import subprocess
import sys
import tempfile
import time

_DEFAULT_OUTPUTS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
_DOCKER_IMAGE = os.environ.get("DS_SANDBOX_IMAGE", "the-pog-sandbox:latest")


def _outputs_dir() -> str:
    """Return the active output directory, respecting DS_OUTPUTS_DIR if set."""
    return os.environ.get("DS_OUTPUTS_DIR", _DEFAULT_OUTPUTS)


def _result(
    output: str,
    error: str,
    success: bool,
    backend: str,
    duration_seconds: float = 0.0,
) -> dict:
    return {
        "output": output.strip(),
        "error": error.strip(),
        "success": success,
        "backend": backend,
        "duration_seconds": round(duration_seconds, 3),
    }


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        probe = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return probe.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _sandbox_preamble() -> str:
    return """
import os, sys
os.makedirs('/workspace/outputs', exist_ok=True)
os.chdir('/workspace')
import matplotlib
matplotlib.use('Agg')
"""


def _local_preamble(outputs_dir: str) -> str:
    return f"""
import os, sys
os.makedirs({repr(outputs_dir)}, exist_ok=True)
os.chdir({repr(outputs_dir)})
os.environ.setdefault('MPLCONFIGDIR', '/tmp/the-pog-matplotlib')
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)
import matplotlib
matplotlib.use('Agg')
"""


def _run_docker(code: str, outputs_dir: str, timeout: int, allow_network: bool) -> dict:
    os.makedirs(outputs_dir, exist_ok=True)
    network = "bridge" if allow_network else "none"
    command = [
        "docker",
        "run",
        "--rm",
        "--interactive",
        "--network",
        network,
        "--memory",
        os.environ.get("DS_SANDBOX_MEMORY", "2g"),
        "--cpus",
        os.environ.get("DS_SANDBOX_CPUS", "1.5"),
        "--pids-limit",
        os.environ.get("DS_SANDBOX_PIDS", "256"),
        "--read-only",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=512m",
        "--security-opt",
        "no-new-privileges",
        "--cap-drop",
        "ALL",
        "--volume",
        f"{os.path.abspath(outputs_dir)}:/workspace/outputs:rw",
        _DOCKER_IMAGE,
    ]
    if hasattr(os, "getuid") and hasattr(os, "getgid"):
        command[2:2] = ["--user", f"{os.getuid()}:{os.getgid()}"]
    try:
        started = time.perf_counter()
        completed = subprocess.run(
            command,
            input=_sandbox_preamble() + "\n" + code,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        diagnostic_output = completed.stdout
        error_output = completed.stderr
        if completed.returncode == 0 and completed.stderr:
            diagnostic_output += "\nSTDERR diagnostics:\n" + completed.stderr
            error_output = ""
        return _result(
            diagnostic_output,
            error_output,
            completed.returncode == 0,
            "docker",
            time.perf_counter() - started,
        )
    except subprocess.TimeoutExpired:
        return _result("", f"Sandbox execution timed out after {timeout}s", False, "docker")


def _run_local(code: str, outputs_dir: str, timeout: int) -> dict:
    """Development-only subprocess backend. This is not a security sandbox."""
    os.makedirs(outputs_dir, exist_ok=True)
    full_code = _local_preamble(outputs_dir) + "\n" + code

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full_code)
        tmp_path = f.name

    try:
        started = time.perf_counter()
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        warning = (
            "WARNING: DS_EXECUTION_BACKEND=local executes generated code without "
            "container isolation.\n"
        )
        diagnostic_output = warning + result.stdout
        error_output = result.stderr
        if result.returncode == 0 and result.stderr:
            diagnostic_output += "\nSTDERR diagnostics:\n" + result.stderr
            error_output = ""
        return _result(
            diagnostic_output,
            error_output,
            result.returncode == 0,
            "local",
            time.perf_counter() - started,
        )
    except subprocess.TimeoutExpired:
        return _result("", f"Code execution timed out after {timeout}s", False, "local")
    finally:
        os.unlink(tmp_path)


def run_code(code: str, timeout: int = 60, allow_network: bool = False) -> dict:
    """Execute generated Python using Docker or an explicit local fallback.

    DS_EXECUTION_BACKEND values:
    - docker: require the hardened Docker sandbox
    - local: development-only subprocess execution
    - auto: use Docker when available, otherwise local with a visible warning
    """
    outputs_dir = _outputs_dir()
    backend = os.environ.get("DS_EXECUTION_BACKEND", "auto").lower()

    if backend not in {"auto", "docker", "local"}:
        return _result(
            "",
            f"Invalid DS_EXECUTION_BACKEND={backend!r}; expected auto, docker, or local",
            False,
            backend,
        )

    docker_ready = _docker_available() if backend in {"auto", "docker"} else False
    if docker_ready:
        return _run_docker(code, outputs_dir, timeout, allow_network)
    if backend == "docker":
        return _result(
            "",
            "Docker sandbox requested but Docker is unavailable. Build the image "
            "with: docker build -f docker/Dockerfile.sandbox -t the-pog-sandbox:latest .",
            False,
            "docker",
        )
    return _run_local(code, outputs_dir, timeout)
