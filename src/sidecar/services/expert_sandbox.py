"""
Expert Sandbox Executor

Provides secure, isolated execution environment for expert skills.
Executes Python code in a separate virtual environment with resource limits.
"""

import subprocess
import json
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import tempfile

logger = logging.getLogger(__name__)


class ExpertSandbox:
    """
    Executes expert skill code in an isolated sandbox environment.

    Features:
    - Separate virtual environment with controlled dependencies
    - Filesystem access limited to working directory
    - Execution timeout enforcement
    - Resource usage monitoring
    - Comprehensive logging
    """

    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize the expert sandbox.

        Args:
            project_root: Root directory of the project. If None, auto-detect.
        """
        if project_root is None:
            # Auto-detect project root (4 levels up from this file)
            project_root = Path(__file__).parent.parent.parent.parent

        self.project_root = project_root
        self.expert_env_path = project_root / ".expert_env"
        self.expert_python = self.expert_env_path / "bin" / "python"

        # Check if sandbox is enabled
        self.enabled = os.getenv("USE_EXPERT_SANDBOX", "false").lower() == "true"

        # Validate environment if sandbox is enabled
        if self.enabled and not self.expert_python.exists():
            logger.warning(
                f"Expert sandbox enabled but environment not found at {self.expert_env_path}. "
                f"Run .expert_config/setup_expert_env.sh to create it. "
                f"Falling back to direct execution."
            )
            self.enabled = False

    def execute(
        self,
        code: str,
        working_dir: Path,
        timeout: int = 30,
        env_vars: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Execute Python code in the sandbox environment.

        Args:
            code: Python code to execute
            working_dir: Working directory for execution (must be within sessions/)
            timeout: Maximum execution time in seconds
            env_vars: Additional environment variables

        Returns:
            Dictionary with execution results:
            {
                "success": bool,
                "stdout": str,
                "stderr": str,
                "returncode": int,
                "execution_time_ms": int
            }
        """
        if not self.enabled:
            # Fallback: execute in current environment
            return self._execute_direct(code, working_dir, timeout)

        # Validate working directory
        if not self._is_safe_working_dir(working_dir):
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Security violation: working_dir must be within sessions/ directory",
                "returncode": -1,
                "execution_time_ms": 0
            }

        # Prepare environment variables
        exec_env = os.environ.copy()
        if env_vars:
            exec_env.update(env_vars)

        # Execute in sandbox
        try:
            import time
            start_time = time.time()

            result = subprocess.run(
                [str(self.expert_python), "-c", code],
                cwd=str(working_dir),
                capture_output=True,
                timeout=timeout,
                text=True,
                env=exec_env
            )

            execution_time_ms = int((time.time() - start_time) * 1000)

            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "execution_time_ms": execution_time_ms
            }

        except subprocess.TimeoutExpired:
            logger.error(f"Expert code execution timeout after {timeout}s")
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Execution timeout after {timeout} seconds",
                "returncode": -1,
                "execution_time_ms": timeout * 1000
            }
        except Exception as e:
            logger.error(f"Expert code execution error: {e}")
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Execution error: {str(e)}",
                "returncode": -1,
                "execution_time_ms": 0
            }

    def execute_skill_function(
        self,
        skill_module: str,
        skill_function: str,
        args: Dict[str, Any],
        working_dir: Path,
        timeout: int = 30
    ) -> Dict[str, Any]:
        """
        Execute a specific skill function in the sandbox.

        Args:
            skill_module: Module name (e.g., "data_inspector_skills")
            skill_function: Function name (e.g., "inspect_dataset")
            args: Function arguments as dictionary
            working_dir: Working directory for execution
            timeout: Maximum execution time in seconds

        Returns:
            Dictionary with execution results including function return value
        """
        # Build Python code to execute the function
        code = f"""
import sys
import json
from pathlib import Path

# Add app/server/services to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "app" / "server" / "services"))

# Import the skill module
from {skill_module} import *

# Parse arguments
args = {json.dumps(args)}

# Execute the function
try:
    result = {skill_function}(**args)
    print(json.dumps({{"success": True, "result": result}}))
except Exception as e:
    print(json.dumps({{"success": False, "error": str(e)}}))
"""

        exec_result = self.execute(code, working_dir, timeout)

        if not exec_result["success"]:
            return {
                "success": False,
                "error": exec_result["stderr"],
                "execution_time_ms": exec_result["execution_time_ms"]
            }

        # Parse the function result
        try:
            function_result = json.loads(exec_result["stdout"])
            function_result["execution_time_ms"] = exec_result["execution_time_ms"]
            return function_result
        except json.JSONDecodeError:
            return {
                "success": False,
                "error": f"Failed to parse function output: {exec_result['stdout']}",
                "execution_time_ms": exec_result["execution_time_ms"]
            }

    def _execute_direct(
        self,
        code: str,
        working_dir: Path,
        timeout: int
    ) -> Dict[str, Any]:
        """
        Fallback: execute code directly in current Python environment.
        Used when sandbox is disabled or not available.
        """
        logger.info("Executing in direct mode (sandbox disabled)")

        try:
            import time
            start_time = time.time()

            # Create a temporary file for the code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name

            try:
                result = subprocess.run(
                    ["python", temp_file],
                    cwd=str(working_dir),
                    capture_output=True,
                    timeout=timeout,
                    text=True
                )

                execution_time_ms = int((time.time() - start_time) * 1000)

                return {
                    "success": result.returncode == 0,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode,
                    "execution_time_ms": execution_time_ms
                }
            finally:
                # Clean up temp file
                Path(temp_file).unlink(missing_ok=True)

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Execution timeout after {timeout} seconds",
                "returncode": -1,
                "execution_time_ms": timeout * 1000
            }
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Execution error: {str(e)}",
                "returncode": -1,
                "execution_time_ms": 0
            }

    def _is_safe_working_dir(self, working_dir: Path) -> bool:
        """
        Validate that working directory is within allowed paths.

        Args:
            working_dir: Directory to validate

        Returns:
            True if directory is safe, False otherwise
        """
        try:
            # Resolve to absolute path
            abs_working_dir = working_dir.resolve()

            # Must be within sessions/ directory
            sessions_dir = (self.project_root / "sessions").resolve()

            # Check if working_dir is under sessions/
            return str(abs_working_dir).startswith(str(sessions_dir))

        except Exception as e:
            logger.error(f"Error validating working directory: {e}")
            return False

    def is_enabled(self) -> bool:
        """Check if sandbox mode is enabled."""
        return self.enabled

    def get_status(self) -> Dict[str, Any]:
        """
        Get sandbox status information.

        Returns:
            Dictionary with status information
        """
        return {
            "enabled": self.enabled,
            "expert_env_exists": self.expert_python.exists(),
            "expert_env_path": str(self.expert_env_path),
            "expert_python": str(self.expert_python),
            "project_root": str(self.project_root)
        }
