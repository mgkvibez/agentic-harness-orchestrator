#!/usr/bin/env bash
# Release gate. Five checks, and the fifth installs the built wheel into a clean
# venv to prove the zero-dependency claim instead of asserting it.
set -uo pipefail
fail() { echo "GATE FAILED: $1"; exit 1; }

echo "== check 1/5: install"
python3 -m pip install -q -e ".[dev]" || fail "install"

echo "== check 2/5: version consistency (pyproject vs module)"
pv=$(python3 - <<'PY'
import pathlib, re
try:
    import tomllib
    print(tomllib.loads(pathlib.Path("pyproject.toml").read_text())["project"]["version"])
except ModuleNotFoundError:
    print(re.search(r'(?m)^version\s*=\s*"([^"]+)"', pathlib.Path("pyproject.toml").read_text()).group(1))
PY
)
mv=$(python3 -c "import agent_harness;print(agent_harness.__version__)")
[ "$pv" = "$mv" ] || fail "pyproject says $pv, module says $mv"

echo "== check 3/5: test suite"
python3 -m pytest tests -q || fail "tests"

echo "== check 4/5: documented CLI exit codes"
aharness demo >/dev/null 2>&1 || fail "demo should exit 0"
aharness models >/dev/null 2>&1 || fail "models should exit 0"
aharness agents >/dev/null 2>&1 || fail "agents should exit 0"
aharness run >/dev/null 2>&1; [ $? -eq 2 ] || fail "run with no task should exit 2"
echo '{"models": ' > /tmp/aho-bad.json
aharness models --models /tmp/aho-bad.json >/dev/null 2>&1; [ $? -eq 2 ] || fail "malformed config should exit 2"
aharness run "Review this codebase" --max-cost 0 >/dev/null 2>&1; [ $? -eq 4 ] || fail "zero budget should exit 4"
if ! python3 -c "import litellm" 2>/dev/null; then
  aharness run "Review this codebase" --live >/dev/null 2>&1; [ $? -eq 5 ] || fail "--live without litellm should exit 5"
fi
aharness trace <(aharness demo --json >/dev/null 2>&1; echo '{"ts":1,"kind":"step_end","step_id":"s1","agent":"a","model":"m","cost_usd":0.001}') >/dev/null 2>&1 || fail "trace summary should read a file"

echo "== check 5/5: wheel in a clean venv (zero third-party dependencies)"
wh=$(mktemp -d)
python3 -m build --wheel --outdir "$wh" -q >/tmp/aho-build.log 2>&1 || { echo "wheel build failed:"; tail -5 /tmp/aho-build.log; exit 1; }
venv=$(mktemp -d)/venv
python3 -m venv "$venv" || fail "venv"
"$venv"/bin/pip install -q "$wh"/*.whl || { echo "wheel install failed"; exit 1; }
"$venv"/bin/aharness demo >/tmp/aho-demo-wheel.out 2>&1 || { cat /tmp/aho-demo-wheel.out; fail "demo from the wheel"; }
grep -q "behind the scenes" /tmp/aho-demo-wheel.out || fail "demo from the wheel printed no trace"
"$venv"/bin/pip list --format=json > /tmp/aho-piplist.json
python3 - <<'PY' || fail "the wheel pulled third-party dependencies"
import json
installed = {p["name"].lower() for p in json.load(open("/tmp/aho-piplist.json"))}
allowed = {"agent-harness-orchestrator", "pip", "setuptools", "wheel"}
extra = installed - allowed
assert not extra, "unexpected dependencies installed: {}".format(sorted(extra))
print("zero third-party dependencies confirmed:", sorted(installed))
PY

echo "ALL CHECKS PASSED"
