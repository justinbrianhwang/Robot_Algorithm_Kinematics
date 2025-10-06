# PyBullet Robotics & UAV Demos

This repository contains lightweight PyBullet demos:
- Two-link planar manipulator
- Manipulator-X style 4-DOF arm with a parallel gripper
- Quad-X UAV (quadcopter)
- SJBot: upper-torso service robot with vacuum demo

Quick start
-----------
Requirements:
- Python 3.9+
- pybullet, numpy

Install:
```
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Examples:
```
python two_link_pybullet_sim.py --gui --traj circle
python manipulator_x_pybullet_sim.py --gui --traj circle
python uav_pybullet_sim.py --gui
python sjbot.py --gui
```

Each script contains in-file comments describing keyboard controls and tuning parameters.
