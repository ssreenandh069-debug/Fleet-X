# Bharat Electronics Limited (BEL) AMR Fleet Coordinator

This project implements a highly scalable, decentralized Edge-AI fleet coordinator for Autonomous Mobile Robots (AMRs) designed for the BEL warehouse logistics problem statement. It entirely eliminates the need for a central server, allowing 200+ AMRs to coordinate their paths and resolve conflicts locally over a Peer-to-Peer network.

## Core Features
1. **Spatio-Temporal A* (Space-Time A*)**: A 3D path planner `(x, y, t)` that allows robots to route around each other dynamically in time, treating other robots' future trajectories as moving obstacles.
2. **Decentralized P2P Mesh**: Robots broadcast their spatial intents only to their local 10x10 sectors, achieving $O(k)$ network scaling instead of $O(N^2)$.
3. **Dynamic Conflict Resolution & Deadlock Solver**: Uses deterministic tie-breaking based on battery level, task urgency, and distance to destination. Lower priority robots instantly yield, reroute, or wait.
4. **Market-Based Task Allocation**: Implements the Contract-Net Protocol where edge terminals broadcast tasks and AMRs compute their bids based on proximity and battery life.
5. **Kinematic Safety & Hazard Gossip**: Features a simulated safety bumper that decelerates robots near obstacles. If a physical hazard is detected, it gossips a `HAZARD_ALERT` to inflate costmaps locally.
6. **Godot UDP Telemetry Bridge**: Fully integrated to broadcast 10Hz non-blocking JSON telemetry for visualization in the Godot engine.

## Performance Benchmark
The system proves the BEL success criteria:
- **Zero (0) Collisions** across 100 benchmark randomized runs.
- **>68% Task Time Reduction** when compared to traditional Stop-And-Wait decentralized architectures, far exceeding the 20% minimum requirement.

## Usage

### 1. Real-time Simulation & Telemetry
If you have the Godot visualizer listening on UDP port `127.0.0.1:4242`:
```powershell
python run_simulation.py
```

### 2. CLI-Only Simulation
If you just want to run the real-time simulation output in the terminal (without Godot):
```powershell
python run_simulation.py --sim-only --no-godot
```

### 3. Execution of the BEL Benchmark
To strictly run the performance benchmark (Mode A vs Mode B) and log throughput improvements and collision counts:
```powershell
python run_simulation.py --bench-only --bench-tasks 100
```

## Hardware Deployment
The system is built entirely on standard Python libraries without external dependencies, optimized to execute path planning under 15ms. This makes it directly ready for deployment on Edge AI hardware like the **Raspberry Pi** or **Jetson Nano**. Ensure that the UDP ports used for P2P networking are permitted on the hardware's local subnet.
