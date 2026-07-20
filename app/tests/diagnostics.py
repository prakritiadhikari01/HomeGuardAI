from __future__ import annotations

import time
import traceback


class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    END = "\033[0m"


class DiagnosticRunner:

    def __init__(self):
        self.total = 0
        self.passed = 0
        self.failed = 0
        self.failures = []

    def banner(self):

        print()
        print("=" * 70)
        print(f"{Colors.BOLD}HOMEGUARD AI ENGINE DIAGNOSTICS{Colors.END}")
        print("=" * 70)

    def section(self, title):

        print()
        print("-" * 70)
        print(f"{Colors.CYAN}{title}{Colors.END}")
        print("-" * 70)

    def run(self, name, func):

        self.total += 1

        start = time.perf_counter()

        try:

            func()

            elapsed = time.perf_counter() - start

            self.passed += 1

            print(
                f"{Colors.GREEN}[PASS]{Colors.END} "
                f"{name:<45}"
                f"{elapsed:.2f}s"
            )

        except Exception as e:

            elapsed = time.perf_counter() - start

            self.failed += 1

            self.failures.append((name, e))

            print(
                f"{Colors.RED}[FAIL]{Colors.END} "
                f"{name:<45}"
                f"{elapsed:.2f}s"
            )

            print()

            traceback.print_exc()

            print()

    def summary(self):

        print()
        print("=" * 70)
        print(f"{Colors.BOLD}SUMMARY{Colors.END}")
        print("=" * 70)

        print(
            f"{Colors.GREEN}Passed : {self.passed}{Colors.END}"
        )

        print(
            f"{Colors.RED}Failed : {self.failed}{Colors.END}"
        )

        print(f"Total  : {self.total}")

        if self.failures:

            print()
            print(f"{Colors.YELLOW}Failures{Colors.END}")

            for name, exc in self.failures:
                print(f" • {name}")
                print(f"   {type(exc).__name__}: {exc}")

        print("=" * 70)