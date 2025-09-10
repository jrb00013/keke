#!/usr/bin/env python3
"""
Comprehensive test runner for Keke Excel Datasheet Tool
"""

import os
import sys
import subprocess
import argparse
import time
import json
from pathlib import Path

class TestRunner:
    """Test runner for Keke project"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.tests_dir = self.project_root / 'tests'
        self.api_dir = self.project_root / 'api'
        self.results = {
            'start_time': time.time(),
            'tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'tests_skipped': 0,
            'coverage': 0,
            'errors': []
        }
    
    def run_command(self, command, capture_output=True):
        """Run a command and return the result"""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=capture_output,
                text=True,
                cwd=self.project_root
            )
            return result
        except Exception as e:
            print(f"Error running command '{command}': {e}")
            return None
    
    def install_dependencies(self):
        """Install test dependencies"""
        print("Installing test dependencies...")
        
        # Install Python dependencies
        pip_result = self.run_command("pip install -r requirements.txt")
        if pip_result and pip_result.returncode != 0:
            print(f"Error installing Python dependencies: {pip_result.stderr}")
            return False
        
        # Install test-specific dependencies
        test_deps = [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.20.0",
            "pytest-cov>=4.0.0",
            "pytest-mock>=3.10.0",
            "pytest-xdist>=3.0.0",
            "pytest-html>=3.1.0",
            "pytest-benchmark>=4.0.0",
            "fastapi>=0.95.0",
            "httpx>=0.24.0",
            "psutil>=5.9.0"
        ]
        
        for dep in test_deps:
            result = self.run_command(f"pip install {dep}")
            if result and result.returncode != 0:
                print(f"Warning: Failed to install {dep}")
        
        # Install Node.js dependencies
        npm_result = self.run_command("npm install")
        if npm_result and npm_result.returncode != 0:
            print(f"Warning: Failed to install Node.js dependencies: {npm_result.stderr}")
        
        return True
    
    def run_unit_tests(self, verbose=False):
        """Run unit tests"""
        print("\n" + "="*60)
        print("RUNNING UNIT TESTS")
        print("="*60)
        
        cmd = "python -m pytest tests/ -m unit"
        
        if verbose:
            cmd += " -v"
        
        cmd += " --tb=short --maxfail=5"
        
        result = self.run_command(cmd)
        
        if result:
            self.results['tests_run'] += self._count_tests_from_output(result.stdout)
            if result.returncode == 0:
                print("✓ Unit tests passed")
            else:
                print("✗ Unit tests failed")
                self.results['errors'].append("Unit tests failed")
                print(result.stdout)
                print(result.stderr)
        
        return result and result.returncode == 0
    
    def run_integration_tests(self, verbose=False):
        """Run integration tests"""
        print("\n" + "="*60)
        print("RUNNING INTEGRATION TESTS")
        print("="*60)
        
        cmd = "python -m pytest tests/ -m integration"
        
        if verbose:
            cmd += " -v"
        
        cmd += " --tb=short --maxfail=3"
        
        result = self.run_command(cmd)
        
        if result:
            self.results['tests_run'] += self._count_tests_from_output(result.stdout)
            if result.returncode == 0:
                print("✓ Integration tests passed")
            else:
                print("✗ Integration tests failed")
                self.results['errors'].append("Integration tests failed")
                print(result.stdout)
                print(result.stderr)
        
        return result and result.returncode == 0
    
    def run_performance_tests(self, verbose=False):
        """Run performance tests"""
        print("\n" + "="*60)
        print("RUNNING PERFORMANCE TESTS")
        print("="*60)
        
        cmd = "python -m pytest tests/ -m slow"
        
        if verbose:
            cmd += " -v"
        
        cmd += " --benchmark-only --benchmark-sort=mean"
        
        result = self.run_command(cmd)
        
        if result:
            self.results['tests_run'] += self._count_tests_from_output(result.stdout)
            if result.returncode == 0:
                print("✓ Performance tests passed")
            else:
                print("✗ Performance tests failed")
                self.results['errors'].append("Performance tests failed")
                print(result.stdout)
                print(result.stderr)
        
        return result and result.returncode == 0
    
    def run_coverage_tests(self):
        """Run tests with coverage analysis"""
        print("\n" + "="*60)
        print("RUNNING COVERAGE ANALYSIS")
        print("="*60)
        
        cmd = "python -m pytest tests/ --cov=api --cov-report=html --cov-report=term-missing"
        
        result = self.run_command(cmd)
        
        if result:
            # Extract coverage percentage
            coverage_line = [line for line in result.stdout.split('\n') if 'TOTAL' in line]
            if coverage_line:
                try:
                    coverage_str = coverage_line[0].split()[-1].replace('%', '')
                    self.results['coverage'] = float(coverage_str)
                except:
                    pass
            
            if result.returncode == 0:
                print("✓ Coverage analysis completed")
                print(f"Coverage: {self.results['coverage']:.1f}%")
            else:
                print("✗ Coverage analysis failed")
                self.results['errors'].append("Coverage analysis failed")
        
        return result and result.returncode == 0
    
    def run_linting(self):
        """Run code linting"""
        print("\n" + "="*60)
        print("RUNNING CODE LINTING")
        print("="*60)
        
        # Python linting
        python_files = list(self.api_dir.glob("*.py"))
        if python_files:
            cmd = f"python -m flake8 {' '.join(str(f) for f in python_files)}"
            result = self.run_command(cmd)
            
            if result:
                if result.returncode == 0:
                    print("✓ Python linting passed")
                else:
                    print("✗ Python linting failed")
                    print(result.stdout)
                    self.results['errors'].append("Python linting failed")
        
        # JavaScript linting
        js_files = list(self.api_dir.glob("*.js"))
        if js_files:
            cmd = f"npx eslint {' '.join(str(f) for f in js_files)}"
            result = self.run_command(cmd)
            
            if result:
                if result.returncode == 0:
                    print("✓ JavaScript linting passed")
                else:
                    print("✗ JavaScript linting failed")
                    print(result.stdout)
                    self.results['errors'].append("JavaScript linting failed")
        
        return True
    
    def run_security_tests(self):
        """Run security tests"""
        print("\n" + "="*60)
        print("RUNNING SECURITY TESTS")
        print("="*60)
        
        # Python security check
        cmd = "python -m bandit -r api/ -f json"
        result = self.run_command(cmd)
        
        if result:
            if result.returncode == 0:
                print("✓ Python security check passed")
            else:
                print("✗ Python security issues found")
                try:
                    security_data = json.loads(result.stdout)
                    high_issues = [issue for issue in security_data['results'] if issue['issue_severity'] == 'HIGH']
                    if high_issues:
                        print(f"Found {len(high_issues)} high severity issues")
                        self.results['errors'].append("Security issues found")
                except:
                    pass
        
        return True
    
    def run_load_tests(self):
        """Run load tests"""
        print("\n" + "="*60)
        print("RUNNING LOAD TESTS")
        print("="*60)
        
        # Start the server in background
        server_process = subprocess.Popen(
            ["python", "api/server.js"],
            cwd=self.project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        try:
            # Wait for server to start
            time.sleep(3)
            
            # Run load tests
            cmd = "python -m pytest tests/test_load.py -v"
            result = self.run_command(cmd)
            
            if result:
                if result.returncode == 0:
                    print("✓ Load tests passed")
                else:
                    print("✗ Load tests failed")
                    self.results['errors'].append("Load tests failed")
        
        finally:
            # Stop the server
            server_process.terminate()
            server_process.wait()
        
        return True
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        print("\n" + "="*60)
        print("GENERATING TEST REPORT")
        print("="*60)
        
        self.results['end_time'] = time.time()
        self.results['duration'] = self.results['end_time'] - self.results['start_time']
        
        # Generate HTML report
        html_report = self._generate_html_report()
        report_file = self.project_root / 'test_report.html'
        
        with open(report_file, 'w') as f:
            f.write(html_report)
        
        print(f"Test report generated: {report_file}")
        
        # Generate JSON report
        json_report = self.project_root / 'test_results.json'
        with open(json_report, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Test results saved: {json_report}")
        
        return True
    
    def _count_tests_from_output(self, output):
        """Count tests from pytest output"""
        lines = output.split('\n')
        test_count = 0
        
        for line in lines:
            if ' passed' in line or ' failed' in line or ' skipped' in line:
                try:
                    count = int(line.split()[0])
                    test_count += count
                except:
                    pass
        
        return test_count
    
    def _generate_html_report(self):
        """Generate HTML test report"""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Keke Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .summary {{ margin: 20px 0; }}
        .error {{ color: red; }}
        .success {{ color: green; }}
        .warning {{ color: orange; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Keke Excel Datasheet Tool - Test Report</h1>
        <p>Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="summary">
        <h2>Test Summary</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Total Tests Run</td>
                <td>{self.results['tests_run']}</td>
            </tr>
            <tr>
                <td>Tests Passed</td>
                <td class="success">{self.results['tests_passed']}</td>
            </tr>
            <tr>
                <td>Tests Failed</td>
                <td class="error">{self.results['tests_failed']}</td>
            </tr>
            <tr>
                <td>Tests Skipped</td>
                <td class="warning">{self.results['tests_skipped']}</td>
            </tr>
            <tr>
                <td>Coverage</td>
                <td>{self.results['coverage']:.1f}%</td>
            </tr>
            <tr>
                <td>Duration</td>
                <td>{self.results['duration']:.2f} seconds</td>
            </tr>
        </table>
    </div>
    
    <div class="errors">
        <h2>Errors and Issues</h2>
        {self._generate_errors_html()}
    </div>
</body>
</html>
        """
        return html
    
    def _generate_errors_html(self):
        """Generate errors HTML section"""
        if not self.results['errors']:
            return "<p class='success'>No errors found!</p>"
        
        html = "<ul>"
        for error in self.results['errors']:
            html += f"<li class='error'>{error}</li>"
        html += "</ul>"
        return html
    
    def run_all_tests(self, verbose=False, coverage=True, lint=True, security=True, performance=True):
        """Run all tests"""
        print("Starting comprehensive test suite for Keke Excel Datasheet Tool")
        print("="*80)
        
        # Install dependencies
        if not self.install_dependencies():
            print("Failed to install dependencies")
            return False
        
        # Run tests
        success = True
        
        if lint:
            success &= self.run_linting()
        
        success &= self.run_unit_tests(verbose)
        success &= self.run_integration_tests(verbose)
        
        if performance:
            success &= self.run_performance_tests(verbose)
        
        if coverage:
            success &= self.run_coverage_tests()
        
        if security:
            success &= self.run_security_tests()
        
        # Generate report
        self.generate_test_report()
        
        # Print summary
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        print(f"Total tests run: {self.results['tests_run']}")
        print(f"Tests passed: {self.results['tests_passed']}")
        print(f"Tests failed: {self.results['tests_failed']}")
        print(f"Tests skipped: {self.results['tests_skipped']}")
        print(f"Coverage: {self.results['coverage']:.1f}%")
        print(f"Duration: {self.results['duration']:.2f} seconds")
        
        if self.results['errors']:
            print(f"\nErrors found: {len(self.results['errors'])}")
            for error in self.results['errors']:
                print(f"  - {error}")
        
        if success:
            print("\n✓ All tests completed successfully!")
        else:
            print("\n✗ Some tests failed!")
        
        return success

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Run Keke test suite')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--no-coverage', action='store_true', help='Skip coverage analysis')
    parser.add_argument('--no-lint', action='store_true', help='Skip linting')
    parser.add_argument('--no-security', action='store_true', help='Skip security tests')
    parser.add_argument('--no-performance', action='store_true', help='Skip performance tests')
    parser.add_argument('--unit-only', action='store_true', help='Run only unit tests')
    parser.add_argument('--integration-only', action='store_true', help='Run only integration tests')
    
    args = parser.parse_args()
    
    runner = TestRunner()
    
    if args.unit_only:
        success = runner.run_unit_tests(args.verbose)
    elif args.integration_only:
        success = runner.run_integration_tests(args.verbose)
    else:
        success = runner.run_all_tests(
            verbose=args.verbose,
            coverage=not args.no_coverage,
            lint=not args.no_lint,
            security=not args.no_security,
            performance=not args.no_performance
        )
    
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
