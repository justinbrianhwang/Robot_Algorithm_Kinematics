"""
SJBot — upper-torso service robot demo with lift and 7-DoF arm

Summary:
- Mobile base + torso lift + 7-DoF manipulator, vacuum cleaner demo (pick up dust particles)
- Camera follow modes, simple particle "dust" generator, and interactive keyboard controls
- Motion and safety limits (slew-rate, joint limits) are enforced

Usage:
  python sjbot.py --gui
  python sjbot.py --headless
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SJBot — Upper-Body + Torso Lift + 7-DoF Arm
모드: 바닥 '먼지' 청소 데모 (손끝=청소기, 바닥 근처를 스치면 먼지 제거)

Usage:
  python sjbot.py --gui
  python sjbot.py --headless

Keys:
  [Base]   W/S: 전후, A/D: 좌/우 이동(스트레이프), Q/E: 회전(요)
  [Torso]  . / , : 승강
  [Arm]    J/L (sh_yaw), I/K (sh_pitch), U/O (sh_roll),
           H/; (elbow), Y/P (wr_pitch), N/M (wr_roll)
  [Vac]    B : 흡입 on/off (기본 on, 손끝이 바닥 가까이 있으면 반경 내 먼지 제거)
  [Debug]  G : EE 프레임 on/off
  [Cam]    C : 카메라 팔로우 on/off, V : 추적 기준(Base↔EE)
           = / - : 줌 인/아웃, 방향키: 궤도 회전, 0 : 카메라 리셋
  [Reset]  R : 먼지 재생성
  [HUD]    F : HUD 토글 (기본 꺼짐)
  [Exit]   ESC : 종료
"""

import math, os, tempfile, time, sys, argparse, random
import numpy as np
import pybullet as p
import pybullet_data

# ---------------- utils ----------------
def make_box(size, pose, rgba=(0.8,0.8,0.8,1), mass=0, lateral_friction=0.8):
    sx, sy, sz = size
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[sx/2, sy/2, sz/2])
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[sx/2, sy/2, sz/2], rgbaColor=rgba)
    bid = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
                            basePosition=pose[0], baseOrientation=pose[1])
    p.changeDynamics(bid, -1, lateralFriction=lateral_friction,
                     rollingFriction=0.0, spinningFriction=0.0)
    return bid

def add_debug_frame(pos, orn, length=0.12, life=0.35):
    R = p.getMatrixFromQuaternion(orn)
    x = np.array([R[0], R[3], R[6]]) * length
    y = np.array([R[1], R[4], R[7]]) * length
    z = np.array([R[2], R[5], R[8]]) * length
    p.addUserDebugLine(pos, (np.array(pos)+x).tolist(), [1,0,0], 3, life)
    p.addUserDebugLine(pos, (np.array(pos)+y).tolist(), [0,1,0], 3, life)
    p.addUserDebugLine(pos, (np.array(pos)+z).tolist(), [0,0,1], 3, life)

# ---------------- camera follow (stabilized) ----------------
def _wrap_angle_deg(a): return (a + 180.0) % 360.0 - 180.0
def _slerp_deg(current, target, alpha):
    d = _wrap_angle_deg(target - current); return current + alpha * d
def _smoothstep01(t):
    t = max(0.0, min(1.0, t)); return t*t*(3 - 2*t)

class CameraFollower:
    def __init__(self, get_target_fn, dist=2.0, yaw=60.0, pitch=-20.0,
                 tau_pos=0.28, tau_orbit=0.20, tau_zoom=0.22,
                 snap_thresh=0.8, transition_dur=0.45, enabled=True):
        self.get_target_fn = get_target_fn; self.enabled = enabled
        self.pos = None; self.yaw = float(yaw); self.pitch = float(pitch); self.dist = float(dist)
        self.des_yaw = float(yaw); self.des_pitch = float(pitch); self.des_dist = float(dist)
        self.tau_pos = float(tau_pos); self.tau_orbit = float(tau_orbit); self.tau_zoom = float(tau_zoom)
        self._snap_thresh = float(snap_thresh); self._tr_dur = float(transition_dur)
        self._tr_t = 0.0; self._tr_start = None; self._tr_end = None
        self._have_prev_raw = False; self._prev_raw = None; self._initd = False
    def set_enabled(self,on:bool): self.enabled = on
    def set_orbit(self,yaw=None,pitch=None,dist=None):
        if yaw is not None: self.des_yaw = float(yaw)
        if pitch is not None: self.des_pitch = float(max(-85.0, min(-5.0, pitch)))
        if dist is not None: self.des_dist = float(max(0.25, dist))
    def nudge_orbit(self,dyaw=0.0,dpitch=0.0,ddist=0.0):
        self.set_orbit(self.des_yaw+dyaw, self.des_pitch+dpitch, self.des_dist+ddist)
    def notify_mode_switch(self):
        raw = np.array(self.get_target_fn())
        self._tr_start = self.pos.copy() if self.pos is not None else raw.copy()
        self._tr_end = raw.copy(); self._tr_t = 0.0
    def _maybe_start_snap_transition(self, raw_target):
        if not self._have_prev_raw:
            self._prev_raw = raw_target.copy(); self._have_prev_raw = True; return
        jump = np.linalg.norm(raw_target - self._prev_raw); self._prev_raw = raw_target.copy()
        if jump > self._snap_thresh:
            self._tr_start = self.pos.copy() if self.pos is not None else raw_target.copy()
            self._tr_end = raw_target.copy(); self._tr_t = 0.0
    def update(self, dt):
        if not self.enabled or p.getConnectionInfo()['connectionMethod'] != p.GUI: return
        dt = max(1e-4, min(0.1, float(dt)))
        raw = np.array(self.get_target_fn())
        if not self._initd:
            self.pos = raw.copy(); self._prev_raw = raw.copy(); self._have_prev_raw = True; self._initd = True
        self._maybe_start_snap_transition(raw)
        if self._tr_start is not None and self._tr_end is not None and self._tr_t < self._tr_dur:
            self._tr_t += dt; s = _smoothstep01(self._tr_t / self._tr_dur)
            target = (1.0 - s) * self._tr_start + s * self._tr_end
        else:
            target = raw.copy(); self._tr_start = self._tr_end = None; self._tr_t = 0.0
        a_pos = 1.0 - math.exp(-dt / self.tau_pos)
        a_orbit = 1.0 - math.exp(-dt / self.tau_orbit)
        a_zoom = 1.0 - math.exp(-dt / self.tau_zoom)
        self.pos = (1.0 - a_pos) * self.pos + a_pos * target
        self.yaw = _slerp_deg(self.yaw, self.des_yaw, a_orbit)
        self.pitch = _slerp_deg(self.pitch, self.des_pitch, a_orbit)
        self.pitch = max(-85.0, min(-5.0, self.pitch))
        self.dist = (1.0 - a_zoom) * self.dist + a_zoom * self.des_dist
        self.dist = max(0.25, self.dist)
        p.resetDebugVisualizerCamera(self.dist, self.yaw, self.pitch, self.pos.tolist())

# ---------------- URDF ----------------
HUMANOID_URDF = """<?xml version="1.0"?>
<robot name="humanoid_server">
  <!-- base -->
  <link name="base">
    <inertial><origin xyz="0 0 0.03"/><mass value="16"/>
      <inertia ixx="0.3184" iyy="0.3184" izz="0.6272"/></inertial>
    <visual><origin xyz="0 0 0.03"/>
      <geometry><cylinder length="0.06" radius="0.28"/></geometry>
      <material name="baseclr"><color rgba="0.2 0.2 0.25 1"/></material></visual>
    <collision><origin xyz="0 0 0.03"/>
      <geometry><cylinder length="0.06" radius="0.28"/></geometry></collision>
  </link>

  <!-- torso lift -->
  <joint name="torso_lift" type="prismatic">
    <parent link="base"/><child link="torso"/>
    <origin xyz="0 0 0.20"/><axis xyz="0 0 1"/>
    <limit lower="0.00" upper="0.25" effort="100" velocity="0.5"/>
  </joint>
  <link name="torso">
    <inertial><origin xyz="0 0 0.15"/><mass value="8"/>
      <inertia ixx="0.0923" iyy="0.1371" izz="0.1093"/></inertial>
    <visual><origin xyz="0 0 0.15"/>
      <geometry><box size="0.34 0.22 0.30"/></geometry>
      <material name="shirt"><color rgba="0.1 0.45 0.85 1"/></material></visual>
    <collision><origin xyz="0 0 0.15"/>
      <geometry><box size="0.34 0.22 0.30"/></geometry></collision>
  </link>

  <!-- head -->
  <joint name="neck_fix" type="fixed"><parent link="torso"/><child link="head"/>
    <origin xyz="0 0 0.33"/></joint>
  <link name="head">
    <inertial><origin xyz="0 0 0.06"/><mass value="2"/>
      <inertia ixx="0.00648" iyy="0.00648" izz="0.00648"/></inertial>
    <visual><origin xyz="0 0 0.06"/>
      <geometry><sphere radius="0.09"/></geometry>
      <material name="skin"><color rgba="1.0 0.84 0.70 1"/></material></visual>
    <collision><origin xyz="0 0 0.06"/><geometry><sphere radius="0.09"/></geometry></collision>
  </link>

  <!-- shoulder yaw -->
  <joint name="sh_yaw" type="revolute">
    <parent link="torso"/><child link="sh_yaw_link"/>
    <origin xyz="0.0 -0.16 0.26"/><axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="40" velocity="2.5"/>
  </joint>
  <link name="sh_yaw_link">
    <inertial><origin xyz="0 0 0"/><mass value="0.3"/>
      <inertia ixx="0.00025" iyy="0.00025" izz="0.000375"/></inertial>
    <visual><origin xyz="0 0 0"/><geometry><cylinder length="0.05" radius="0.05"/></geometry>
      <material name="jnt"><color rgba="0.2 0.2 0.2 1"/></material></visual>
    <collision><origin xyz="0 0 0"/><geometry><cylinder length="0.05" radius="0.05"/></geometry></collision>
  </link>

  <!-- shoulder pitch -->
  <joint name="sh_pitch" type="revolute">
    <parent link="sh_yaw_link"/><child link="sh_pitch_link"/>
    <origin xyz="0 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-2.5" upper="2.5" effort="40" velocity="2.5"/>
  </joint>
  <link name="sh_pitch_link">
    <inertial><origin xyz="0.11 0 0"/><mass value="0.6"/>
      <inertia ixx="0.00048" iyy="0.00266" izz="0.00266"/></inertial>
    <visual><origin xyz="0.11 0 0" rpy="0 1.5708 0"/><geometry><capsule length="0.22" radius="0.04"/></geometry>
      <material name="arm1"><color rgba="0.1 0.45 0.85 1"/></material></visual>
    <collision><origin xyz="0.11 0 0" rpy="0 1.5708 0"/><geometry><capsule length="0.22" radius="0.04"/></geometry></collision>
  </link>

  <!-- shoulder roll -->
  <joint name="sh_roll" type="revolute">
    <parent link="sh_pitch_link"/><child link="upper_arm"/>
    <origin xyz="0.22 0 0"/><axis xyz="1 0 0"/>
    <limit lower="-2.5" upper="2.5" effort="30" velocity="3.0"/>
  </joint>
  <link name="upper_arm">
    <inertial><origin xyz="0.18 0 0"/><mass value="0.8"/>
      <inertia ixx="0.00058" iyy="0.00893" izz="0.00893"/></inertial>
    <visual><origin xyz="0.18 0 0" rpy="0 1.5708 0"/><geometry><capsule length="0.36" radius="0.038"/></geometry>
      <material name="arm2"><color rgba="0.1 0.45 0.85 1"/></material></visual>
    <collision><origin xyz="0.18 0 0" rpy="0 1.5708 0"/><geometry><capsule length="0.36" radius="0.038"/></geometry></collision>
  </link>

  <!-- elbow -->
  <joint name="elbow" type="revolute">
    <parent link="upper_arm"/><child link="forearm"/>
    <origin xyz="0.36 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-2.6" upper="2.6" effort="30" velocity="3.0"/>
  </joint>
  <link name="forearm">
    <inertial><origin xyz="0.20 0 0"/><mass value="0.7"/>
      <inertia ixx="0.00043" iyy="0.00951" izz="0.00951"/></inertial>
    <visual><origin xyz="0.20 0 0" rpy="0 1.5708 0"/><geometry><capsule length="0.40" radius="0.035"/></geometry>
      <material name="arm3"><color rgba="0.1 0.45 0.85 1"/></material></visual>
    <collision><origin xyz="0.20 0 0" rpy="0 1.5708 0"/><geometry><capsule length="0.40" radius="0.035"/></geometry></collision>
  </link>

  <!-- wrist pitch -->
  <joint name="wr_pitch" type="revolute">
    <parent link="forearm"/><child link="wr_pitch_link"/>
    <origin xyz="0.40 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-2.5" upper="2.5" effort="20" velocity="3.5"/>
  </joint>
  <link name="wr_pitch_link">
    <inertial><origin xyz="0.03 0 0"/><mass value="0.2"/>
      <inertia ixx="4e-5" iyy="8e-5" izz="8e-5"/></inertial>
    <visual><origin xyz="0.03 0 0"/><geometry><box size="0.06 0.035 0.035"/></geometry>
      <material name="jointbox"><color rgba="0.2 0.2 0.2 1"/></material></visual>
    <collision><origin xyz="0.03 0 0"/><geometry><box size="0.06 0.035 0.035"/></geometry></collision>
  </link>

  <!-- wrist roll (tool: vacuum nozzle) -->
  <joint name="wr_roll" type="revolute">
    <parent link="wr_pitch_link"/><child link="tool"/>
    <origin xyz="0.06 0 0"/><axis xyz="1 0 0"/>
    <limit lower="-3.14" upper="3.14" effort="15" velocity="4.0"/>
  </joint>
  <link name="tool">
    <inertial><origin xyz="0.06 0 0"/><mass value="0.18"/>
      <inertia ixx="2.6e-5" iyy="1.5e-4" izz="1.5e-4"/></inertial>
    <visual>
      <origin xyz="0.06 0 0"/><geometry><box size="0.12 0.05 0.035"/></geometry>
      <material name="nozzle"><color rgba="0.05 0.05 0.05 1"/></material>
    </visual>
    <collision><origin xyz="0.06 0 0"/><geometry><box size="0.12 0.05 0.035"/></geometry></collision>
  </link>
</robot>
"""

def write_urdf(text, name="humanoid_server.urdf"):
    path = os.path.join(tempfile.gettempdir(), name)
    with open(path, "w") as f: f.write(text)
    return path

# ---------------- dust ----------------
def make_dust_particle(xy, radius=0.015, z=0.01):
    col = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=0.006)
    vis = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=0.006,
                              rgbaColor=(0.12,0.12,0.12,1.0))
    bid = p.createMultiBody(baseMass=0.001, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
                            basePosition=[xy[0], xy[1], z],
                            baseOrientation=p.getQuaternionFromEuler([math.pi/2,0,0]))
    p.changeDynamics(bid, -1, lateralFriction=0.9, rollingFriction=0.0, spinningFriction=0.0,
                     restitution=0.0, linearDamping=0.02, angularDamping=0.02)
    return bid

def scatter_dust(area_center, area_half_ext, num=150, z=0.008, seed=7):
    rng = np.random.default_rng(seed)
    dust_ids = []
    cx, cy = area_center; hx, hy = area_half_ext
    for _ in range(num):
        x = float(rng.uniform(cx - hx, cx + hx))
        y = float(rng.uniform(cy - hy, cy + hy))
        r = float(rng.uniform(0.010, 0.022))
        dust_ids.append(make_dust_particle((x,y), radius=r, z=z))
    return dust_ids

# ---------------- environment ----------------
def build_classroom():
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")

    desk_size = (0.8, 0.6, 0.75)
    origin = np.array([0.0, 0.0, 0.0])
    cell = np.array([1.2, 1.0, 0])

    desks = {}
    for r in range(4):
        for c in range(2):
            if r==0 and c==0: continue
            xy = origin + np.array([0.8 + c*cell[0], -0.8 - r*cell[1], 0])
            z = desk_size[2]/2
            desk_id = make_box(
                desk_size,
                ((xy+[0,0,z]).tolist(), p.getQuaternionFromEuler([0,0,0])),
                (0.7,0.5,0.3,1), mass=0
            )
            p.changeDynamics(desk_id, -1, lateralFriction=1.1,
                             rollingFriction=0.01, spinningFriction=0.01,
                             restitution=0.0)
            desks[(r,c)] = (desk_id, xy)

    # 칠판(시각용 얇은 보드)
    board_size = (3.2, 0.04, 1.0)
    board_pos  = (1.6, -0.45, 1.10)
    _board = make_box(board_size, (board_pos, p.getQuaternionFromEuler([0,0,0])),
                      (0.05,0.35,0.20,1.0), mass=0)
    _frame = make_box((3.24, 0.045, 0.04),
                      ((board_pos[0], board_pos[1], board_pos[2]+board_size[2]/2+0.02),
                       p.getQuaternionFromEuler([0,0,0])),
                      (0.7,0.55,0.35,1), mass=0)

    # ✅ 칠판 투과 방지용 '보이지 않는 충돌 패널'(바닥~머리 높이)
    # 시각은 투명, 충돌만 두껍게 (두께 0.08m)
    panel_size = (3.3, 0.08, 1.3)                 # x폭, y두께, z높이
    panel_pos  = (board_pos[0], board_pos[1], 0.65)  # 바닥~1.3m
    _panel = make_box(panel_size, (panel_pos, p.getQuaternionFromEuler([0,0,0])),
                      (0.0,0.0,0.0,0.0), mass=0, lateral_friction=1.2)

    # 바닥 먼지 흩뿌리기: 교실 중심 근처 넓은 영역
    area_center = (0.9, -1.25)
    area_half   = (1.4, 1.8)
    dust_ids = scatter_dust(area_center, area_half, num=220, z=0.006, seed=11)

    return dict(
        desks=desks,
        dust_ids=dust_ids, dust_area_center=area_center, dust_area_half=area_half
    )

# ---------------- SJBot ----------------
class SJBot:
    def __init__(self):
        self.urdf_path=write_urdf(HUMANOID_URDF)
        self.robot=p.loadURDF(self.urdf_path, basePosition=[-0.3,-0.6,0.05],
                              baseOrientation=p.getQuaternionFromEuler([0,0,0]),
                              flags=p.URDF_USE_INERTIA_FROM_FILE)
        self.jmap={p.getJointInfo(self.robot,j)[1].decode():j for j in range(p.getNumJoints(self.robot))}
        self.j_torso="torso_lift"
        self.j_arm=["sh_yaw","sh_pitch","sh_roll","elbow","wr_pitch","wr_roll"]

        # torso
        self.torso_pos=0.10
        p.setJointMotorControl2(self.robot,self.jmap[self.j_torso],p.POSITION_CONTROL,
                                targetPosition=self.torso_pos,force=300,maxVelocity=0.5)

        # arm stiff
        self.qdes={nm:p.getJointState(self.robot,self.jmap[nm])[0] for nm in self.j_arm}
        for nm in self.j_arm:
            jid=self.jmap[nm]
            p.setJointMotorControl2(self.robot,jid,p.POSITION_CONTROL,
                targetPosition=self.qdes[nm],force=180.0,positionGain=0.9,velocityGain=1.0)

        # base with slew-rate limiting
        self.vx=self.vy=self.wz=0.0
        self.vx_cmd=self.vy_cmd=self.wz_cmd=0.0
        self.ax_max=1.8    # m/s^2
        self.aw_max=5.0    # rad/s^2

    # base
    def set_base_cmd(self,vx,vy,wz):
        self.vx_cmd, self.vy_cmd, self.wz_cmd = vx, vy, wz

    def step_base(self,dt):
        def slew(cur, cmd, amax):
            dv = np.clip(cmd - cur, -amax*dt, amax*dt)
            return cur + dv
        self.vx = slew(self.vx, self.vx_cmd, self.ax_max)
        self.vy = slew(self.vy, self.vy_cmd, self.ax_max)
        self.wz = slew(self.wz, self.wz_cmd, self.aw_max)

        pos,orn=p.getBasePositionAndOrientation(self.robot)
        yaw=p.getEulerFromQuaternion(orn)[2]+self.wz*dt
        c,s=math.cos(yaw),math.sin(yaw)
        dx_b,dy_b=self.vx*dt,self.vy*dt
        dx_w,dy_w=c*dx_b-s*dy_b,s*dx_b+c*dy_b
        new_pos=[pos[0]+dx_w,pos[1]+dy_w,pos[2]]
        new_orn=p.getQuaternionFromEuler([0,0,yaw])
        p.resetBasePositionAndOrientation(self.robot,new_pos,new_orn)
        p.resetBaseVelocity(self.robot,[0,0,0],[0,0,0])

    # arm
    def nudge(self,nm,delta):
        jid=self.jmap[nm]
        lo,hi=p.getJointInfo(self.robot,jid)[8],p.getJointInfo(self.robot,jid)[9]
        raw=float(np.clip(self.qdes[nm]+delta,lo,hi))
        alpha=0.45
        self.qdes[nm]=alpha*raw+(1-alpha)*self.qdes[nm]
    def arm_position_update(self,dt):
        RATE=2.5
        for nm in self.j_arm:
            jid=self.jmap[nm]
            q_now=p.getJointState(self.robot,jid)[0]
            dq_allow=RATE*dt
            q_next=float(np.clip(self.qdes[nm],q_now-dq_allow,q_now+dq_allow))
            p.setJointMotorControl2(self.robot,jid,p.POSITION_CONTROL,
                targetPosition=q_next,force=180.0,
                positionGain=0.9,velocityGain=1.0,maxVelocity=1.6)

    # ee pose
    def ee_pose(self):
        tool_idx=self.jmap["wr_roll"]
        st=p.getLinkState(self.robot,tool_idx,computeForwardKinematics=True)
        return st[4],st[5]

# ---------------- helper: suction (vacuum) ----------------
def vacuum_cleanup(env, bot, radius=0.18, floor_z=0.02, ee_floor_max=0.15, max_remove_per_step=30, draw=False):
    ee_p,_ = bot.ee_pose()
    if ee_p[2] > ee_floor_max: 
        return 0
    dust_ids = env["dust_ids"]
    if not dust_ids: return 0
    ex, ey = ee_p[0], ee_p[1]
    r2 = radius*radius
    removed = 0
    keep = []
    for did in dust_ids:
        ppos, _ = p.getBasePositionAndOrientation(did)
        dx = ppos[0]-ex; dy = ppos[1]-ey
        if dx*dx + dy*dy <= r2:
            p.removeBody(did)
            removed += 1
            if removed >= max_remove_per_step:
                keep.extend([d for d in dust_ids if d not in keep and d != did])
                break
        else:
            keep.append(did)
    env["dust_ids"] = keep
    if draw and p.getConnectionInfo()['connectionMethod']==p.GUI:
        p.addUserDebugLine([ex,ey,floor_z+0.001],[ex,ey,floor_z+0.001],
                           [0.9,0.1,0.1], lineWidth=6, lifeTime=0.12)
    return removed

# ---------------- main ----------------
def main():
    ap=argparse.ArgumentParser()
    g=ap.add_mutually_exclusive_group()
    g.add_argument("--gui", action="store_true", help="GUI mode")
    g.add_argument("--headless", action="store_true", help="Direct/headless mode")
    args=ap.parse_args()

    cid = p.connect(p.GUI if args.gui or not args.headless else p.DIRECT)
    if cid < 0:
        print("PyBullet connect failed"); sys.exit(1)

    # 시각화 옵션
    p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)

    # 물리
    p.setTimeStep(1/300)
    try:
        p.setPhysicsEngineParameter(
            numSubSteps=8, numSolverIterations=240,
            useSplitImpulse=1, splitImpulsePenetrationThreshold=-0.04,
            enableConeFriction=1, contactBreakingThreshold=1e-3,
            allowedCcdPenetration=5e-4, deterministicOverlappingPairs=1
        )
    except TypeError:
        p.setPhysicsEngineParameter(numSubSteps=8, numSolverIterations=240, enableConeFriction=1)
    p.setGravity(0,0,-9.81)

    env=build_classroom()
    bot=SJBot()

    # 초기 정착
    for _ in range(150): p.stepSimulation()

    # HUD
    SHOW_HUD = False
    hud_id = None
    def show_hud(cleaned=0, remain=None, vac_on=True):
        msg = (f"[Base] W/S fwd/back | A/D strafe | Q/E yaw | "
               f"[Torso] . , | [Arm] J/L I/K U/O H/; Y/P N/M | "
               f"[Vac] B toggle ({'ON' if vac_on else 'OFF'}) | R reset | ESC quit | F HUD\n"
               f"Dust cleaned: {cleaned} | Remaining: {remain if remain is not None else len(env['dust_ids'])}")
        return p.addUserDebugText(msg, [-1.9,0.9,1.5],[0,0,0],1.05, lifeTime=0)
    if SHOW_HUD and p.getConnectionInfo()['connectionMethod']==p.GUI:
        hud_id = show_hud(0, len(env["dust_ids"]))

    # 카메라
    cam_mode = 'base'
    def cam_target_base():
        pos,_ = p.getBasePositionAndOrientation(bot.robot)
        return (np.array(pos) + np.array([0.,0.,0.25]))
    def cam_target_ee():
        ee_p,_ = bot.ee_pose()
        return (np.array(ee_p) + np.array([0.,0.,0.10]))
    get_target = lambda: (cam_target_base() if cam_mode=='base' else cam_target_ee())
    cam = CameraFollower(get_target, enabled=True)

    last=time.time()
    step_count=0
    draw_frames=True
    vacuum_on = True
    cleaned_total = 0

    dust_area_center = env["dust_area_center"]; dust_area_half = env["dust_area_half"]

    running=True
    while running:
        now=time.time(); dt=now-last; last=now
        keys=p.getKeyboardEvents()
        def down(ch): oc=ord(ch); return oc in keys and (keys[oc]&p.KEY_IS_DOWN)
        def pressed(ch): oc=ord(ch); return oc in keys and (keys[oc]&p.KEY_WAS_TRIGGERED)

        # exit
        if 27 in keys and (keys[27] & p.KEY_WAS_TRIGGERED):  # ESC
            running=False

        # HUD 토글
        if pressed('f') or pressed('F'):
            if hud_id is not None:
                p.removeUserDebugItem(hud_id); hud_id=None
            else:
                if p.getConnectionInfo()['connectionMethod']==p.GUI:
                    hud_id = show_hud(cleaned_total, len(env["dust_ids"]), vacuum_on)

        # ---------------- Base control (요청대로 수정) ----------------
        vx=vy=wz=0.0
        if down('w') or down('W'): vx = 0.9
        if down('s') or down('S'): vx = -0.9
        if down('a') or down('A'): vy = +0.9     # 좌로 스트레이프
        if down('d') or down('D'): vy = -0.9     # 우로 스트레이프 (바디 좌표계 기준)
        if down('q') or down('Q'): wz += 1.8     # 반시계(좌회전)
        if down('e') or down('E'): wz -= 1.8     # 시계(우회전)
        bot.set_base_cmd(vx,vy,wz); bot.step_base(dt)

        # Torso (그대로)
        if hasattr(bot, 'torso_pos'):
            if down('.'):
                jid=bot.jmap["torso_lift"]; lo,hi=p.getJointInfo(bot.robot,jid)[8],p.getJointInfo(bot.robot,jid)[9]
                bot.torso_pos=float(np.clip(bot.torso_pos+0.004,lo,hi))
                p.setJointMotorControl2(bot.robot,jid,p.POSITION_CONTROL,
                                        targetPosition=bot.torso_pos,maxVelocity=0.5,force=300)
            if down(','):
                jid=bot.jmap["torso_lift"]; lo,hi=p.getJointInfo(bot.robot,jid)[8],p.getJointInfo(bot.robot,jid)[9]
                bot.torso_pos=float(np.clip(bot.torso_pos-0.004,lo,hi))
                p.setJointMotorControl2(bot.robot,jid,p.POSITION_CONTROL,
                                        targetPosition=bot.torso_pos,maxVelocity=0.5,force=300)

        # Arm (그대로)
        step=0.02
        if any(down(k) for k in "ZXCVBNMIUOHJKLP;"): step=0.008
        keymap={'j':("sh_yaw",-1),'l':("sh_yaw",+1),
                'i':("sh_pitch",-1),'k':("sh_pitch",+1),
                'u':("sh_roll",-1),'o':("sh_roll",+1),
                'h':("elbow",+1),';':("elbow",-1),
                'y':("wr_pitch",+1),'p':("wr_pitch",-1),
                'n':("wr_roll",-1),'m':("wr_roll",+1)}
        for k,(jn,sgn) in keymap.items():
            if down(k) or down(k.upper()): bot.nudge(jn,sgn*step)
        bot.arm_position_update(dt)

        # Vacuum toggle
        if pressed('b') or pressed('B'):
            vacuum_on = not vacuum_on
            if hud_id is not None:
                p.removeUserDebugItem(hud_id); hud_id = show_hud(cleaned_total, len(env["dust_ids"]), vacuum_on)

        # Vacuum cleanup
        if vacuum_on:
            removed = vacuum_cleanup(env, bot, radius=0.18, floor_z=0.0, ee_floor_max=0.15, max_remove_per_step=40, draw=False)
            if removed:
                cleaned_total += removed
                if hud_id is not None:
                    p.removeUserDebugItem(hud_id); hud_id = show_hud(cleaned_total, len(env["dust_ids"]), vacuum_on)

        # reset: 먼지 재생성
        if pressed('r') or pressed('R'):
            for did in env["dust_ids"]:
                try: p.removeBody(did)
                except: pass
            env["dust_ids"] = scatter_dust(dust_area_center, dust_area_half, num=220, z=0.006, seed=random.randint(0,9999))
            cleaned_total = 0
            if hud_id is not None:
                p.removeUserDebugItem(hud_id); hud_id = show_hud(cleaned_total, len(env["dust_ids"]), vacuum_on)

        # debug frame toggle
        if pressed('g') or pressed('G'):
            draw_frames = not draw_frames

        # camera controls
        if pressed('c') or pressed('C'):
            cam.set_enabled(not cam.enabled)
        if pressed('v') or pressed('V'):
            cam_mode = 'ee' if cam_mode=='base' else 'base'
            cam.notify_mode_switch()
        if pressed('=') or pressed('+'):  cam.nudge_orbit(ddist=-0.2)
        if pressed('-') or pressed('_'):  cam.nudge_orbit(ddist=+0.2)
        KEY_LEFT = p.B3G_LEFT_ARROW; KEY_RIGHT = p.B3G_RIGHT_ARROW
        KEY_UP = p.B3G_UP_ARROW; KEY_DOWN = p.B3G_DOWN_ARROW
        if KEY_LEFT in keys and (keys[KEY_LEFT] & p.KEY_WAS_TRIGGERED):  cam.nudge_orbit(dyaw=-8.0)
        if KEY_RIGHT in keys and (keys[KEY_RIGHT] & p.KEY_WAS_TRIGGERED): cam.nudge_orbit(dyaw=+8.0)
        if KEY_UP in keys and (keys[KEY_UP] & p.KEY_WAS_TRIGGERED):       cam.nudge_orbit(dpitch=+6.0)
        if KEY_DOWN in keys and (keys[KEY_DOWN] & p.KEY_WAS_TRIGGERED):   cam.nudge_orbit(dpitch=-6.0)
        if pressed('0'): cam.set_orbit(yaw=60.0, pitch=-20.0, dist=2.0)

        # debug frame (스로틀링)
        step_count += 1
        if draw_frames and (p.getConnectionInfo()['connectionMethod']==p.GUI) and (step_count % 10 == 0):
            ee_p,ee_q=bot.ee_pose()
            add_debug_frame(ee_p,ee_q,0.1,0.35)

        p.stepSimulation(); cam.update(dt)
        time.sleep(max(0.0, (1/300) - (time.time()-now)))

    p.disconnect()

if __name__=="__main__":
    main()
