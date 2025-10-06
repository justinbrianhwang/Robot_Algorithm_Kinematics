# PyBullet Robotics & UAV Demos

This repository contains lightweight PyBullet demos:
- Two-link planar manipulator

  <img width="521" height="296" alt="image" src="https://github.com/user-attachments/assets/5b7a851d-4a20-4366-8799-6759011d7343" />

- Manipulator-X style 4-DOF arm with a parallel gripper

  <img width="515" height="284" alt="image" src="https://github.com/user-attachments/assets/9964b389-a70a-4928-909f-e0c2cf9d94bd" />

- Quad-X UAV (quadcopter)

  <img width="559" height="276" alt="image" src="https://github.com/user-attachments/assets/e5382385-8b95-43c8-b9d7-2a191c27f180" />

- SJBot: upper-torso service robot with vacuum demo

  <img width="685" height="443" alt="image" src="https://github.com/user-attachments/assets/493a8d5d-16ff-4ed7-921c-72d43a2aa98a" />


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
