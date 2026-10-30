"""Exercise the thermal guard without using the real GPU or datasets."""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

SOURCE = Path(__file__).parents[1] / "scripts" / "run_fedhca2_restart_s0.sh"
GENERIC_SOURCE = Path(__file__).parents[1] / "scripts" / "thermal_supervise.sh"
HANDOFF_SOURCE = Path(__file__).parents[1] / "scripts" / "wait_and_run_remaining.sh"


class TestRestartSupervisor(unittest.TestCase):
    def fixture(self, root, initial_temp=49, load_temp=95):
        (root / "scripts").mkdir()
        shutil.copyfile(SOURCE, root / "scripts/run_fedhca2_restart_s0.sh")
        (root / ".venv/bin").mkdir(parents=True)
        worker = root / ".venv/bin/python"
        worker.write_text("#!/bin/bash\nsleep 30\n")
        worker.chmod(0o755)
        binary = root / "bin"
        binary.mkdir()
        smi = binary / "nvidia-smi"
        smi.write_text(
            "#!/bin/bash\n"
            "case \"$*\" in\n"
            " *--query-compute-apps*) exit 0;;\n"
            f" *--query-gpu=temperature.gpu*) echo {initial_temp};;\n"
            f" *) echo '2026-10-30 00:00:00, {load_temp}, 100, 100, 100, 200, 300, Active';;\n"
            "esac\n"
        )
        smi.chmod(0o755)
        logs = root / "logs"
        logs.mkdir()
        env = dict(os.environ, PATH=f"{binary}:{os.environ['PATH']}")
        return root / "scripts/run_fedhca2_restart_s0.sh", logs, env

    def test_thermal_threshold_stops_only_the_test_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script, logs, env = self.fixture(root)
            result = subprocess.run(["bash", str(script), str(logs)], cwd=root,
                                    env=env, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 75, result.stderr)
            self.assertEqual((logs / "exit_code").read_text().strip(), "75")
            self.assertIn("Thermal safeguard", result.stderr)

    def test_hot_gpu_is_rejected_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script, logs, env = self.fixture(root, initial_temp=93)
            result = subprocess.run(["bash", str(script), str(logs)], cwd=root,
                                    env=env, capture_output=True, text=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((logs / "worker.pid").exists())

    def test_followup_guard_stops_test_worker_when_hot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, logs, env = self.fixture(root)
            script = root / "scripts/thermal_supervise.sh"
            shutil.copyfile(GENERIC_SOURCE, script)
            worker = root / ".venv/bin/python"
            result = subprocess.run(["bash", str(script), str(logs), str(worker)], cwd=root,
                                    env=env, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 75, result.stderr)
            self.assertEqual((logs / "exit_code").read_text().strip(), "75")

    def test_handoff_refuses_failed_control(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scripts").mkdir()
            script = root / "scripts/wait_and_run_remaining.sh"
            shutil.copyfile(HANDOFF_SOURCE, script)
            logs = root / "control_logs"
            logs.mkdir()
            (logs / "exit_code").write_text("75\n")
            result = subprocess.run(["bash", str(script), "999999999", str(logs)], cwd=root,
                                    capture_output=True, text=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("no follow-up jobs started", result.stderr)


if __name__ == "__main__":
    unittest.main()
