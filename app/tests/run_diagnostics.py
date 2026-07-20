from __future__ import annotations

from app.tests.diagnostics import DiagnosticRunner

# Import individual diagnostic modules
from app.tests.test_runtime import runtime_tests
from app.tests.test_camera import camera_tests
from app.tests.test_motion import motion_tests


def main():

    runner = DiagnosticRunner()

    runner.banner()

    runner.section("Runtime")
    runtime_tests(runner)

    runner.section("Camera")
    camera_tests(runner)

    runner.section("Motion")
    motion_tests(runner)

    # Future Parts
    #
    # runner.section("YOLO")
    # yolo_tests(runner)
    #
    # runner.section("Tracking")
    # tracking_tests(runner)
    #
    # runner.section("Pipeline")
    # pipeline_tests(runner)
    #
    # runner.section("Django")
    # django_tests(runner)
    #
    # runner.section("Qwen")
    # qwen_tests(runner)

    runner.summary()


if __name__ == "__main__":
    main()