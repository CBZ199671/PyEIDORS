import subprocess


def check_tmp_filesystem():
    """Check filesystem info for /tmp with a cross-platform fallback."""
    try:
        result = subprocess.run(
            ["df", "-T", "/tmp"], capture_output=True, text=True, check=False
        )
        output = result.stdout + result.stderr
        command = "df -T /tmp"
        if result.returncode != 0:
            result = subprocess.run(
                ["df", "/tmp"], capture_output=True, text=True, check=False
            )
            output = result.stdout + result.stderr
            command = "df /tmp"

        if "tmpfs" in output.lower():
            print(
                "/tmp uses tmpfs (memory filesystem) - extremely fast IO performance!"
            )
        else:
            print("/tmp uses regular disk filesystem")
        print(output)
        return {"command": command, "output": output, "returncode": result.returncode}
    except Exception:
        return None


def test_check_tmp_filesystem_runs():
    """Smoke test to keep the cache diagnostic script pytest-runnable."""
    info = check_tmp_filesystem()
    assert info is not None
    assert "output" in info
