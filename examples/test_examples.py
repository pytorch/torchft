import os
import subprocess
import tempfile
import time
import socket
from pathlib import Path

import pytest


def get_example_directories():
    """Get all example directories that have .torchxconfig files."""
    current_dir = Path.cwd()
    if current_dir.name == "examples":
        examples_root = current_dir
    else:
        examples_root = Path("examples")
    
    return [
        d for d in examples_root.iterdir() 
        if d.is_dir() and (d / ".torchxconfig").exists()
    ]

@pytest.fixture(scope="session")
def lighthouse_server():
    """Start a lighthouse server for the tests."""
    default_port = 29510
    lighthouse_url = f"http://localhost:{default_port}"
    
    # Kill any existing process using the lighthouse port
    try:
        result = subprocess.run(["lsof", "-ti", f":{default_port}"], capture_output=True, text=True)
        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                subprocess.run(["kill", pid], capture_output=True)
            time.sleep(1)  # Give time for cleanup
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    lighthouse_proc = subprocess.Popen(
        ["torchft_lighthouse", 
        "--min_replicas", "1", 
        "--quorum_tick_ms", "100", 
        "--join_timeout_ms", "10000", 
        "--bind", f"[::]:{default_port}"
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    
    time.sleep(1)
    
    yield lighthouse_url
    
    lighthouse_proc.terminate()
    try:
        lighthouse_proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        lighthouse_proc.kill()
        lighthouse_proc.wait()

class TestTorchXExamples:
    """Test that torchx run works for all examples."""

    @pytest.mark.parametrize("example_dir", get_example_directories())
    def test_training_script_exists(self, example_dir):
        """Test that the training script referenced in config exists."""
        config_path = example_dir / ".torchxconfig"
        
        import configparser
        config = configparser.ConfigParser()
        config.read(config_path)
        
        # Find component sections
        for section_name in config.sections():
            if section_name.startswith("component:"):
                section = config[section_name]
                if "script" in section:
                    script_path = example_dir / section["script"]
                    assert script_path.exists(), f"Missing training script {script_path} referenced in {config_path}"

    @pytest.mark.parametrize("example_dir", get_example_directories())
    def test_torchx_config_valid(self, example_dir):
        """Test that .torchxconfig files are valid."""
        config_path = example_dir / ".torchxconfig"
        assert config_path.exists(), f"Missing .torchxconfig in {example_dir}"
        
        # Try to parse the config
        import configparser
        config = configparser.ConfigParser()
        config.read(config_path)
        
        # Should have cli:run section
        assert "cli:run" in config.sections(), f"Missing [cli:run] section in {config_path}"
        
        # Should have component reference
        cli_section = config["cli:run"]
        assert "component" in cli_section, f"Missing component in [cli:run] section of {config_path}"
        
        component_ref = cli_section["component"]
        assert ":" in component_ref, f"Invalid component reference format in {config_path}: {component_ref}"

    @pytest.mark.parametrize("example_dir", get_example_directories())
    def test_torchx_run_quick(self, example_dir, lighthouse_server):
        """Test that torchx run works with QUICK_RUN for each example."""

        timeout_seconds = 120
        self._test_example(example_dir, lighthouse_server, timeout_seconds)
    
    def _test_example(self, example_dir, lighthouse_server, timeout_seconds=120):
        """Test regular examples (non-ddp_proactive)."""
        print(f"\n=== Testing {example_dir.name} ===")
        
        # Set environment for quick run with memory management

        env = os.environ.copy()
        env["QUICK_RUN"] = "1"
        env["TORCHFT_LIGHTHOUSE"] = lighthouse_server
        # Enable PyTorch memory management features to prevent fragmentation
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        if example_dir.name == "ddp_proactive":
            env["TORCHFT_PROACTIVE_RECOVERY"] = "1"
        else:
            env["TORCHFT_PROACTIVE_RECOVERY"] = "0" 
            
        cmd = ["torchx", "run"]

        # Start the process with a new process group for better cleanup control
        try:
            result = subprocess.run(
                cmd,
                cwd=example_dir,
                env=env,
                timeout=timeout_seconds,  # Increased timeout to allow for optimized sync frequencies
                capture_output=False,  # Let training logs print to console
                text=True,
                preexec_fn=os.setsid,  # Create new process group for easier cleanup
            )
            
            # Training should complete successfully
            # When capture_output=False, stdout/stderr are None, so we just check return code
            assert result.returncode == 0, f"torchx run failed for {example_dir} with return code {result.returncode}"
            
            print("-" * 30)
            print(f"✅ {example_dir.name} completed successfully! ")
            print("-" * 30)
            
        except subprocess.TimeoutExpired:
            # If timeout occurs, try to clean up any remaining processes
            print(f"Test timed out for {example_dir.name}, cleaning up processes...")
            self._cleanup_training_processes()
            raise
        except Exception as e:
            # On any other error, also try cleanup
            print(f"Test failed for {example_dir.name}: {e}, cleaning up processes...")
            self._cleanup_training_processes()
            raise
        finally:
            # Always attempt cleanup after each test to prevent accumulation
            time.sleep(1)  # Give processes time to naturally terminate
            self._cleanup_training_processes()
    
    def _cleanup_training_processes(self):
        """Clean up any remaining training processes to prevent GPU memory accumulation."""
        try:
            # Find and kill any remaining training processes
            result = subprocess.run(
                ["ps", "aux"], capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                pids_to_kill = []
                
                for line in lines:
                    # Look for training scripts and torchrun processes
                    if any(pattern in line for pattern in ['train_diloco.py', 'train_localsgd.py', 'train_ddp_proactive.py', 'torchrun']):
                        parts = line.split()
                        if len(parts) > 1:
                            try:
                                pid = int(parts[1])
                                pids_to_kill.append(pid)
                            except (ValueError, IndexError):
                                continue
                
                # Kill the processes
                for pid in pids_to_kill:
                    try:
                        subprocess.run(["kill", "-9", str(pid)], capture_output=True, timeout=5)
                    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
                        pass  # Process might already be dead
                        
                if pids_to_kill:
                    print(f"🧹 Cleaned up {len(pids_to_kill)} stale training processes")
                    
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            # If cleanup fails, log but don't crash the test
            print("Process cleanup encountered issues, but continuing...")
            pass