#!/usr/bin/env python3
"""
PyEIDORS Comprehensive Test Suite
Runs all available tests including basic module tests, functional tests, and performance tests.
"""

import subprocess
import sys
import os
from pathlib import Path
import time


class TestRunner:
    """Test runner class."""

    def __init__(self):
        self.results = {}
        self.start_time = time.time()

    def run_test(self, test_name: str, test_script: str) -> bool:
        """
        Run a single test script.

        Args:
            test_name: Test name.
            test_script: Test script path.

        Returns:
            Whether the test passed.
        """
        print(f"\n{'='*60}")
        print(f"Running: {test_name}")
        print(f"Script: {test_script}")
        print(f"{'='*60}")

        if not Path(test_script).exists():
            print(f"❌ Test script not found: {test_script}")
            self.results[test_name] = {'status': 'missing', 'time': 0}
            return False

        start_time = time.time()

        try:
            # Run test script
            result = subprocess.run(
                [sys.executable, test_script],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            test_time = time.time() - start_time

            if result.returncode == 0:
                print(f"✅ {test_name} test passed")
                self.results[test_name] = {'status': 'passed', 'time': test_time}

                # Show last few lines of output if available
                if result.stdout:
                    output_lines = result.stdout.strip().split('\n')
                    print("📋 Test output summary:")
                    for line in output_lines[-5:]:  # Show last 5 lines
                        print(f"   {line}")

                return True
            else:
                print(f"❌ {test_name} test failed (exit code: {result.returncode})")
                self.results[test_name] = {'status': 'failed', 'time': test_time}

                # Show error messages
                if result.stderr:
                    print("🔍 Error messages:")
                    error_lines = result.stderr.strip().split('\n')
                    for line in error_lines[-10:]:  # Show last 10 error lines
                        print(f"   {line}")

                return False

        except subprocess.TimeoutExpired:
            test_time = time.time() - start_time
            print(f"⏰ {test_name} test timed out")
            self.results[test_name] = {'status': 'timeout', 'time': test_time}
            return False

        except Exception as e:
            test_time = time.time() - start_time
            print(f"💥 {test_name} test execution error: {e}")
            self.results[test_name] = {'status': 'error', 'time': test_time}
            return False

    def print_summary(self):
        """Print test summary."""
        total_time = time.time() - self.start_time

        print(f"\n{'='*80}")
        print("🏆 PyEIDORS Test Suite Complete")
        print(f"{'='*80}")

        print(f"\n📊 Test Results:")
        print(f"{'Test Name':<30} {'Status':<10} {'Time(s)':<10}")
        print(f"{'-'*55}")

        passed = failed = timeout = error = missing = 0

        for test_name, result in self.results.items():
            status = result['status']
            test_time = result['time']

            if status == 'passed':
                status_emoji = "✅ Passed"
                passed += 1
            elif status == 'failed':
                status_emoji = "❌ Failed"
                failed += 1
            elif status == 'timeout':
                status_emoji = "⏰ Timeout"
                timeout += 1
            elif status == 'error':
                status_emoji = "💥 Error"
                error += 1
            else:
                status_emoji = "❓ Missing"
                missing += 1

            print(f"{test_name:<30} {status_emoji:<10} {test_time:<10.2f}")

        total_tests = len(self.results)

        print(f"\n📈 Overall Statistics:")
        print(f"   Total tests: {total_tests}")
        print(f"   Passed: {passed}")
        print(f"   Failed: {failed}")
        print(f"   Timeout: {timeout}")
        print(f"   Error: {error}")
        print(f"   Missing: {missing}")
        print(f"   Success rate: {passed/total_tests*100:.1f}%" if total_tests > 0 else "   Success rate: 0%")
        print(f"   Total time: {total_time:.2f} seconds")

        # Provide suggestions
        print(f"\n💡 Suggestions:")
        if failed > 0:
            print("   - Check failed tests, may need to fix dependencies or configuration issues")
        if timeout > 0:
            print("   - Timed out tests may need performance optimization or increased timeout")
        if error > 0:
            print("   - Tests with errors need code debugging")
        if missing > 0:
            print("   - Missing test scripts need to be created")
        if passed == total_tests:
            print("   - 🎉 All tests passed! System is working properly.")

        return passed == total_tests


def main():
    """Main function."""
    print("🚀 Starting PyEIDORS Comprehensive Test Suite")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")

    runner = TestRunner()

    # Define test list
    tests = [
        ("Basic Module Test", "test_pyeidors.py"),
        ("Simplified System Test", "test_simplified_eit_system.py"),
        ("Complete System Test", "test_complete_eit_system.py"),
    ]

    all_passed = True

    # Run all tests
    for test_name, test_script in tests:
        success = runner.run_test(test_name, test_script)
        if not success:
            all_passed = False

    # Print summary
    runner.print_summary()

    # Create test report
    create_test_report(runner.results)

    return all_passed


def create_test_report(results):
    """Create detailed test report."""
    output_dir = Path("test_results")
    output_dir.mkdir(exist_ok=True)

    report_file = output_dir / "test_report.md"

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# PyEIDORS Test Report\n\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Test Results\n\n")
        f.write("| Test Name | Status | Time(s) |\n")
        f.write("|----------|------|----------|\n")

        for test_name, result in results.items():
            status = result['status']
            test_time = result['time']

            status_map = {
                'passed': '✅ Passed',
                'failed': '❌ Failed',
                'timeout': '⏰ Timeout',
                'error': '💥 Error',
                'missing': '❓ Missing'
            }

            status_text = status_map.get(status, status)
            f.write(f"| {test_name} | {status_text} | {test_time:.2f} |\n")

        f.write("\n## System Information\n\n")
        f.write(f"- Python version: {sys.version}\n")
        f.write(f"- Platform: {sys.platform}\n")
        f.write(f"- Working directory: {os.getcwd()}\n")

        f.write("\n## Module Status\n\n")
        try:
            import pyeidors
            env = pyeidors.check_environment()
            f.write(f"- FEniCS: {'✅' if env['fenics_available'] else '❌'}\n")
            f.write(f"- PyTorch: {'✅' if env['torch_available'] else '❌'}\n")
            f.write(f"- CUDA: {'✅' if env['cuda_available'] else '❌'}\n")
            f.write(f"- CUQIpy: {'✅' if env['cuqi_available'] else '❌'}\n")
            if env['torch_available']:
                f.write(f"- PyTorch version: {env['torch_version']}\n")
                f.write(f"- GPU count: {env['cuda_device_count']}\n")
        except Exception as e:
            f.write(f"- Environment check failed: {e}\n")

    print(f"\n📄 Detailed test report saved to: {report_file.absolute()}")


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
