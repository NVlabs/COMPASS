#!/usr/bin/env python3

# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


class CompassContainer:
    """Simplified container interface for Compass ROS2 navigation deployment."""

    # Container configuration constants
    COMPASS_CONTAINER_NAME = "compass-ros2-navigator"
    ISAAC_SIM_CONTAINER_NAME = "isaac-sim"

    def __init__(self, ros_domain_id=0, rmw_implementation="rmw_fastrtps_cpp", enable_rviz=False):
        self.base_dir = Path(__file__).resolve().parent
        self.ros_domain_id = ros_domain_id
        self.rmw_implementation = rmw_implementation
        self.enable_rviz = enable_rviz

        # File paths
        self.compass_compose_file = self.base_dir / "docker" / "docker-compose.compass.yaml"
        self.isaac_compose_file = self.base_dir / "docker" / "docker-compose.isaacsim.yaml"
        self.fastdds_config_path = self.base_dir / "docker" / ".ros" / "fastdds.xml"

        # Set COMPASS_PATH if not set (used by build_compass_docker.sh)
        if not os.environ.get("COMPASS_PATH"):
            os.environ["COMPASS_PATH"] = str(self.base_dir.parent)

    def _run_command(self, cmd, check=True):
        """Run a shell command and handle errors."""
        try:
            result = subprocess.run(cmd, check=check, cwd=self.base_dir)
            return result.returncode == 0
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def _is_container_running(self, container_name):
        """Check if a Docker container is running by name."""
        check_cmd = ["docker", "ps", "-q", "-f", f"name={container_name}"]
        result = subprocess.run(check_cmd, capture_output=True, text=True, check=False)
        return bool(result.stdout.strip())

    def _container_exists(self, container_name):
        """Check if a Docker container exists (running or stopped) by name."""
        check_cmd = ["docker", "ps", "-a", "-q", "-f", f"name={container_name}"]
        result = subprocess.run(check_cmd, capture_output=True, text=True, check=False)
        return bool(result.stdout.strip())

    def _convert_model(self, force_conversion=False):
        """Handle TensorRT model conversion with caching."""
        if force_conversion:
            print("[INFO] Force TensorRT conversion - regenerating engine...")
            rm_cmd = [
                "docker", "exec", self.COMPASS_CONTAINER_NAME, "rm", "-f",
                "/tmp/engines/compass_carter.engine"
            ]
            self._run_command(rm_cmd, check=False)

        # Check if TensorRT engine already exists
        check_cmd = [
            "docker", "exec", self.COMPASS_CONTAINER_NAME, "test", "-f",
            "/tmp/engines/compass_carter.engine"
        ]
        engine_exists = subprocess.run(check_cmd, capture_output=True, check=False).returncode == 0

        if engine_exists and not force_conversion:
            print("[INFO] TensorRT engine already exists, skipping conversion")
            return True

        print("[INFO] Converting ONNX to TensorRT engine (1-5 minutes)...")
        conv_cmd = [
            "docker", "exec", self.COMPASS_CONTAINER_NAME, "python3",
            "/usr/local/bin/trt_conversion.py", "--onnx-path", "/tmp/models/compass_carter.onnx",
            "--trt-path", "/tmp/engines/compass_carter.engine"
        ]

        if not self._run_command(conv_cmd, check=False):
            print("[ERROR] TensorRT conversion failed")
            return False

        print("[INFO] TensorRT engine created successfully")
        return True

    def _check_display_and_x11(self):
        """Check if DISPLAY is set and X11 permissions are properly configured."""
        # Check if DISPLAY is set
        display = os.environ.get('DISPLAY')
        if not display:
            print("[ERROR] DISPLAY environment variable not set")
            print("        Set with: export DISPLAY=:0")
            print("        Then run: xhost +local:docker")
            return False

        # Check xhost permissions
        try:
            result = subprocess.run(['xhost'], capture_output=True, text=True, check=False)
            if result.returncode == 0:
                output = result.stdout.strip()
                if any(x in output
                       for x in ['LOCAL:', 'SI:localuser:docker', 'access control disabled']):
                    return True
                else:
                    print("[ERROR] Docker containers do not have X11 access")
                    print("        Run: xhost +local:docker")
                    return False
            else:
                print("[ERROR] Failed to check xhost permissions")
                return False
        except FileNotFoundError:
            print("[ERROR] xhost command not found")
            print("        Make sure you're running from a graphical environment")
            return False

    def start(self, force_trt_conversion=False, enable_rviz=None):
        """Start the Compass container in detached mode."""
        # Use instance setting if not explicitly overridden
        use_rviz = enable_rviz if enable_rviz is not None else self.enable_rviz

        gui_status = 'enabled' if use_rviz else 'disabled'
        print(f"[INFO] Starting Compass container (ROS_DOMAIN_ID={self.ros_domain_id}, "
              f"GUI={gui_status})")

        # Only check DISPLAY and X11 permissions if RViz is enabled
        if use_rviz and not self._check_display_and_x11():
            return False

        # Check if container already exists (running or stopped)
        if self._container_exists(self.COMPASS_CONTAINER_NAME):
            if self._is_container_running(self.COMPASS_CONTAINER_NAME):
                print(f"[ERROR] Container '{self.COMPASS_CONTAINER_NAME}' is already running")
                print("[INFO] Use './compass_container.py enter' to access it")
            else:
                print(f"[ERROR] Container '{self.COMPASS_CONTAINER_NAME}' exists but is stopped")
                print("[INFO] Use './compass_container.py stop' to remove it, then start again")
            return False

        if not self.compass_compose_file.exists():
            print(f"[ERROR] Compose file not found: {self.compass_compose_file}")
            print("[INFO] Run './build_compass_docker.sh' first")
            return False

        cmd = [
            "docker",
            "compose",
            "-f",
            str(self.compass_compose_file),
            "run",
            "-d",
            "--name",
            self.COMPASS_CONTAINER_NAME,
            "-e",
            f"ROS_DOMAIN_ID={self.ros_domain_id}",
            "-e",
            f"RMW_IMPLEMENTATION={self.rmw_implementation}",
        ]

        # Only add DISPLAY environment variable if RViz is enabled
        if use_rviz:
            cmd.extend(["-e", f"DISPLAY={os.environ.get('DISPLAY', '')}"])

        cmd.append("compass-ros2-deploy")

        if not self._run_command(cmd):
            return False

        print(f"[INFO] Container '{self.COMPASS_CONTAINER_NAME}' started")
        return self._convert_model(force_trt_conversion)

    def enter(self, force_trt_conversion=False, enable_rviz=None):
        """Enter the running Compass container."""
        if not self._is_container_running(self.COMPASS_CONTAINER_NAME):
            print(f"[ERROR] Container '{self.COMPASS_CONTAINER_NAME}' is not running")
            print("[INFO] Use './compass_container.py start' to start it first")
            return False

        if force_trt_conversion and not self._convert_model(force_trt_conversion):
            print("[WARNING] TensorRT conversion failed, continuing anyway...")

        use_rviz = enable_rviz if enable_rviz is not None else self.enable_rviz
        rviz_flag = "enable_rviz:=true" if use_rviz else "enable_rviz:=false"

        print("[INFO] Entering container. Available commands:")
        print(f"  - launch_compass.sh {rviz_flag}        # Start navigator")
        print("  - ros2 topic list                      # List ROS2 topics")
        print("  - exit                                  # Exit container")

        cmd = ["docker", "exec", "-it", self.COMPASS_CONTAINER_NAME, "bash"]
        try:
            subprocess.run(cmd, check=False)
        except KeyboardInterrupt:
            print("\n[INFO] Exited container")

    def stop(self):
        """Stop and remove the Compass container."""
        print("[INFO] Stopping Compass container...")

        if not self._container_exists(self.COMPASS_CONTAINER_NAME):
            print(f"[INFO] Container '{self.COMPASS_CONTAINER_NAME}' does not exist")
            return True

        self._run_command(["docker", "stop", self.COMPASS_CONTAINER_NAME], check=False)
        if self._run_command(["docker", "rm", self.COMPASS_CONTAINER_NAME], check=False):
            print(f"[INFO] Container '{self.COMPASS_CONTAINER_NAME}' stopped")
            return True
        return False

    def logs(self):
        """Show container logs."""
        cmd = ["docker", "logs", "-f", self.COMPASS_CONTAINER_NAME]
        try:
            subprocess.run(cmd, check=False)
        except KeyboardInterrupt:
            print("\n[INFO] Stopped following logs")

    def status(self):
        """Show container status."""
        check_cmd = [
            "docker", "ps", "-a", "-f", f"name={self.COMPASS_CONTAINER_NAME}", "--format",
            "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
        ]
        result = subprocess.run(check_cmd, capture_output=True, text=True, check=False)

        if self.COMPASS_CONTAINER_NAME in result.stdout:
            print(result.stdout)
        else:
            print(f"[INFO] Container '{self.COMPASS_CONTAINER_NAME}' does not exist")

    def launch_nav(self, force_trt_conversion=False, enable_rviz=None):
        """Quick launch: start container and launch compass navigator."""
        use_rviz = enable_rviz if enable_rviz is not None else self.enable_rviz

        gui_status = 'enabled' if use_rviz else 'disabled'
        print(f"[INFO] Quick launch: starting container and launching navigator "
              f"(GUI={gui_status})...")

        # Check if container is already running
        if self._is_container_running(self.COMPASS_CONTAINER_NAME):
            print(f"[INFO] Container '{self.COMPASS_CONTAINER_NAME}' is already running, using it")
            # Handle TensorRT conversion if requested
            if force_trt_conversion:
                if not self._convert_model(force_trt_conversion):
                    print("[WARNING] TensorRT conversion failed, continuing anyway...")
        else:
            # Container not running, start it first
            if not self.start(force_trt_conversion=force_trt_conversion, enable_rviz=use_rviz):
                return False
            time.sleep(2)    # Wait for container to be ready

        print("[INFO] Launching Compass navigator...")
        cmd = ["docker", "exec", self.COMPASS_CONTAINER_NAME, "launch_compass.sh"]

        # Add rviz parameter to launch command
        rviz_flag = "enable_rviz:=true" if use_rviz else "enable_rviz:=false"
        cmd.append(rviz_flag)

        return self._run_command(cmd, check=False)

    def launch_isaac_sim(self):
        """Launch Isaac Sim container with ROS2 bridge enabled."""
        print(f"[INFO] Launching Isaac Sim (ROS_DOMAIN_ID={self.ros_domain_id})")

        # Check if Isaac Sim container is already running
        if self._is_container_running(self.ISAAC_SIM_CONTAINER_NAME):
            print(f"[WARNING] Isaac Sim container '{self.ISAAC_SIM_CONTAINER_NAME}' "
                  f"is already running")
            print("[INFO] Close Isaac Sim GUI to stop the container, or run: "
                  "docker stop isaac-sim")
            return False

        # Check DISPLAY and X11 permissions
        if not self._check_display_and_x11():
            return False

        # Check FastDDS config
        if not self.fastdds_config_path.exists():
            print(f"[ERROR] FastDDS config not found: {self.fastdds_config_path}")
            return False

        cmd = [
            "docker", "compose", "-f",
            str(self.isaac_compose_file), "run", "--name", self.ISAAC_SIM_CONTAINER_NAME, "--rm",
            "-e", f"ROS_DOMAIN_ID={self.ros_domain_id}", "-e",
            f"RMW_IMPLEMENTATION={self.rmw_implementation}", "-v",
            f"{self.fastdds_config_path.absolute()}:/root/.ros/fastdds.xml:ro", "isaac-sim"
        ]

        print("[INFO] Press Ctrl+C to stop Isaac Sim (container will auto-remove)")
        try:
            subprocess.run(cmd, check=False)
        except KeyboardInterrupt:
            print("\n[INFO] Isaac Sim stopped")

        return True

    def debug_network(self):
        """Debug ROS2 network setup by checking environment variables in both containers."""
        print("[INFO] Debugging ROS2 network setup...")
        print("=" * 60)

        # Check both containers
        containers = [(self.COMPASS_CONTAINER_NAME, "🔍 Compass Navigator Container"),
                      (self.ISAAC_SIM_CONTAINER_NAME, "🔍 Isaac Sim Container")]

        for container_name, title in containers:
            print(f"\n{title}:")
            print("-" * 40)

            if self._is_container_running(container_name):
                print(f"Container name: {container_name}")
                env_check_cmd = [
                    "docker", "exec", container_name, "bash", "-c",
                    "echo 'ROS_DOMAIN_ID: '$ROS_DOMAIN_ID && "
                    "echo 'RMW_IMPLEMENTATION: '$RMW_IMPLEMENTATION && "
                    "echo 'FASTRTPS_DEFAULT_PROFILES_FILE: '$FASTRTPS_DEFAULT_PROFILES_FILE && "
                    "echo 'User: '$(whoami) && "
                    "echo 'FastDDS config exists: '"
                    "$(test -f \"$FASTRTPS_DEFAULT_PROFILES_FILE\" && echo 'YES' || echo 'NO')"
                ]
                try:
                    subprocess.run(env_check_cmd, check=True, text=True)
                except subprocess.CalledProcessError:
                    print(f"[ERROR] Failed to check {container_name} environment")
            else:
                print(f"[WARNING] Container '{container_name}' is not running")

        # ROS2 topic test
        if self._is_container_running(self.COMPASS_CONTAINER_NAME):
            print("\n🔍 ROS2 Topic Discovery Test:")
            topic_cmd = [
                "docker", "exec", self.COMPASS_CONTAINER_NAME, "bash", "-c",
                "source /opt/ros/humble/setup.bash && timeout 10 ros2 topic list"
            ]

            result = subprocess.run(topic_cmd, capture_output=True, text=True, check=False)
            if result.returncode == 0:
                topics = result.stdout.strip().split('\n')
                isaac_topics = [
                    t for t in topics
                    if any(kw in t for kw in ['chassis', 'cmd_vel', 'stereo', 'odom'])
                ]

                print(f"Found {len(topics)} ROS2 topics")
                if isaac_topics:
                    print("✅ Isaac Sim topics detected:")
                    for topic in isaac_topics[:5]:
                        print(f"  - {topic}")
                    if len(isaac_topics) > 5:
                        print(f"  ... and {len(isaac_topics) - 5} more")
                else:
                    print("❌ No Isaac Sim topics found - make sure Isaac Simulation "
                          "is running by hitting play button")
            else:
                print("❌ Failed to list ROS2 topics (timeout or ROS2 not ready)")
        else:
            print("\n[WARNING] Cannot test ROS2 topics - compass container not running")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compass ROS2 Navigation Container Manager",
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog="""
Examples:
  ./compass_container.py start                    # Start headless container
  ./compass_container.py start --rviz             # Start with RViz GUI
  ./compass_container.py enter                    # Enter container
  ./compass_container.py launch_nav               # Quick headless launch
  ./compass_container.py launch_nav --rviz        # Quick launch with GUI
  ./compass_container.py launch_isaac_sim         # Launch Isaac Sim
  ./compass_container.py debug_network            # Debug ROS2 setup
  ./compass_container.py stop                     # Stop container

ROS2 Configuration:
  ./compass_container.py start --ros-domain-id 5  # Custom domain ID
  ./compass_container.py launch_nav --rviz --ros-domain-id 5  # GUI + custom domain
""")

    parser.add_argument("--ros-domain-id", type=int, default=0, help="ROS2 Domain ID (default: 0)")
    parser.add_argument("--rmw-implementation",
                        default="rmw_fastrtps_cpp",
                        help="ROS2 middleware (default: rmw_fastrtps_cpp)")
    parser.add_argument("--rviz",
                        action="store_true",
                        help="Enable RViz GUI (default: False, headless mode)")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Commands with TensorRT conversion option
    for cmd_name, help_text in [("start", "Start the Compass container"),
                                ("enter", "Enter the running container"),
                                ("launch_nav", "Quick launch: start + run navigator")]:
        cmd_parser = subparsers.add_parser(cmd_name, help=help_text)
        cmd_parser.add_argument("--force-trt-conversion",
                                action="store_true",
                                help="Force TensorRT engine regeneration")

    # Simple commands
    for cmd_name, help_text in [("stop", "Stop and remove the container"),
                                ("logs", "Show container logs"),
                                ("status", "Show container status"),
                                ("launch_isaac_sim", "Launch Isaac Sim with ROS2 bridge"),
                                ("debug_network", "Debug ROS2 network setup")]:
        subparsers.add_parser(cmd_name, help=help_text)

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    compass = CompassContainer(ros_domain_id=args.ros_domain_id,
                               rmw_implementation=args.rmw_implementation,
                               enable_rviz=args.rviz)

    success = True
    force_trt = getattr(args, 'force_trt_conversion', False)
    enable_rviz = args.rviz

    if args.command == "start":
        success = compass.start(force_trt_conversion=force_trt, enable_rviz=enable_rviz)
    elif args.command == "enter":
        success = compass.enter(force_trt_conversion=force_trt, enable_rviz=enable_rviz)
    elif args.command == "stop":
        success = compass.stop()
    elif args.command == "logs":
        compass.logs()
    elif args.command == "status":
        compass.status()
    elif args.command == "launch_nav":
        success = compass.launch_nav(force_trt_conversion=force_trt, enable_rviz=enable_rviz)
    elif args.command == "launch_isaac_sim":
        success = compass.launch_isaac_sim()
    elif args.command == "debug_network":
        compass.debug_network()
    else:
        print(f"[ERROR] Unknown command: {args.command}")
        success = False

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
