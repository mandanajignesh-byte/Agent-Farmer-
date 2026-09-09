"""Run main.py the way Kaggle does, before submitting.

    python check_submission.py

A submission is a bare main.py dropped into /kaggle_simulations/agent/ with no
sibling modules on the path. Local testing does not catch that: main.py and
market.py sat side by side here and imported fine, then the submission died
with ModuleNotFoundError on the first turn.

Worse, kaggle-environments swallows a per-turn exception and substitutes a
default action, so a broken agent can report DONE with a plausible-looking
score rather than an error. This copies main.py alone into an empty directory,
strips the project from sys.path, and fails loudly on any exception.
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROBE = '''
import sys, traceback
sys.path = [p for p in sys.path if "Ep-1" not in p]
from kaggle_environments import make

errors = []
import main as agent_module
original = agent_module.agent

def watched(obs):
    try:
        return original(obs)
    except Exception:
        errors.append(traceback.format_exc())
        raise

agent_module.agent = watched
env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": 7})
env.run([watched, "random"])
final = env.steps[-1]
print("STATUS", [s.status for s in final])
print("REWARD", [s.reward for s in final])
print("ERRORS", len(errors))
if errors:
    print(errors[0][-800:])
'''


def main():
    source = Path("main.py").resolve()
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy(source, Path(tmp) / "main.py")
        (Path(tmp) / "probe.py").write_text(PROBE)
        result = subprocess.run(
            [sys.executable, "probe.py"], cwd=tmp, capture_output=True, text=True
        )

    out = result.stdout
    print(out.strip() or result.stderr[-1500:])

    ok = "ERRORS 0" in out and "'DONE', 'DONE'" in out
    print("\nSUBMISSION OK" if ok else "\nSUBMISSION WOULD FAIL - do not upload")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
