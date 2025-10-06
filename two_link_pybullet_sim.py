"""
Two-link planar manipulator — minimal, stable PyBullet demo (expand/contract hotkeys included)

Key points:
- Uses only POSITION_CONTROL (no torque/velocity modes) — stable even on Linux with llvmpipe
- F: extend to the maximum radius (Rmax) in the current shoulder direction and hold
- B: contract to the minimum radius (Rmin) in the same direction and hold
- S: smooth handover back to the base trajectory (circle/lem) over 0.6 seconds from the current end-effector pose
- Debug drawing rate-limited to reduce "User debug draw failed" warnings
- Keyboard input uses edge-triggering only (KEY_WAS_TRIGGERED)

Usage:
  python two_link_pybullet_sim.py --gui --traj circle

See in-file comments for full key mappings and tuning parameters.
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two-link planar manipulator — minimal, stable PyBullet demo (확장/수축 핫키 포함)

핵심:
- POSITION_CONTROL만 사용(토크/속도 모드 제거) → 리눅스/llvmpipe에서도 안정
- F: 현재 어깨 각도 방향으로 **최대 반경 Rmax**까지 확장(쫙 펴기, hold)
- B: 같은 방향으로 **최소 반경 Rmin**까지 수축(접기, hold)
- S: 현재 EE 위치에서 기본 궤적(circle/lem)으로 **0.6초 부드럽게 복귀**(handover)
- 디버그 라인(EE 트레이스/타깃) **생성률 제한**으로 ‘User debug draw failed’ 경고 완화
- 키 입력은 **엣지 트리거(KEY_WAS_TRIGGERED)**만 사용(길게 눌러도 1회만 처리)

Usage:
  python two_link_pybullet_sim.py --gui --traj circle
  python two_link_pybullet_sim.py --gui --traj lem

Keys:
  F/B : 확장/수축 시작(도달 후 유지)
  S   : 수동 종료 후 0.6초 스무스 복귀
  E   : Elbow up/down IK 가지 토글
  +/- : 속도 스케일 업/다운
  P/R/C : 일시정지/리셋/트레이스 지우기
"""

import argparse, math, time, tempfile
from pathlib import Path
from collections import deque
from typing import Tuple

import numpy as np
import pybullet as p
import pybullet_data

# ---------------------------------
# Helpers
# ---------------------------------

def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def write_tmp(text: str, name: str) -> str:
    path = Path(tempfile.gettempdir()) / name
    path.write_text(text)
    return str(path)


# ---------------------------------
# Cross-platform key helper (edge only)
# ---------------------------------
EDGE = p.KEY_WAS_TRIGGERED  # treat key on press only

def key_edge(keys, *codes):
    return any((c in keys) and (keys[c] & EDGE) for c in codes)

K_F = [ord('f'), ord('F')]
K_B = [ord('b'), ord('B')]
K_S = [ord('s'), ord('S')]
K_E = [ord('e'), ord('E')]
K_P = [ord('p'), ord('P')]
K_R = [ord('r'), ord('R')]
K_C = [ord('c'), ord('C')]
K_PLUS  = [ord('+'), ord('='), 43, 61]
K_MINUS = [ord('-'), 45]


# ---------------------------------
# URDF (planar 2R)
# ---------------------------------

def two_link_urdf(L1: float, L2: float, payload: float = 0.0) -> str:
    m1, m2, mee = 0.3, 0.25, max(0.01, payload)
    w = h = 0.03

    def box_I(m, lx, ly, lz):
        ixx = (1/12)*m*(ly*ly + lz*lz)
        iyy = (1/12)*m*(lx*lx + lz*lz)
        izz = (1/12)*m*(lx*lx + ly*ly)
        return ixx, iyy, izz

    ixx1, iyy1, izz1 = box_I(m1, L1, w, h)
    ixx2, iyy2, izz2 = box_I(m2, L2, w, h)
    ixxe, iyye, izze = box_I(mee, 0.04, 0.04, 0.02)

    return f"""
<?xml version='1.0'?>
<robot name='two_link_planar_min'>
  <link name='base'>
    <inertial><mass value='1.0'/>
      <inertia ixx='1e-3' iyy='1e-3' izz='1e-3' ixy='0' ixz='0' iyz='0'/>
    </inertial>
    <visual>
      <origin xyz='0 0 0.015' rpy='0 0 0'/>
      <geometry><cylinder radius='0.06' length='0.03'/></geometry>
    </visual>
  </link>

  <joint name='joint1' type='revolute'>
    <parent link='base'/>
    <child link='link1'/>
    <origin xyz='0 0 0.03' rpy='0 0 0'/>
    <axis xyz='0 0 1'/>
    <dynamics damping='0.04' friction='0.02'/>
    <limit lower='-{math.pi}' upper='{math.pi}' effort='40' velocity='8'/>
  </joint>

  <link name='link1'>
    <inertial><mass value='{m1}'/>
      <inertia ixx='{ixx1}' iyy='{iyy1}' izz='{izz1}' ixy='0' ixz='0' iyz='0'/>
    </inertial>
    <visual>
      <origin xyz='{L1/2} 0 0.03' rpy='0 0 0'/>
      <geometry><box size='{L1} {w} {h}'/></geometry>
    </visual>
    <collision>
      <origin xyz='{L1/2} 0 0.03' rpy='0 0 0'/>
      <geometry><box size='{L1} {w} {h}'/></geometry>
    </collision>
  </link>

  <joint name='joint2' type='revolute'>
    <parent link='link1'/>
    <child link='link2'/>
    <origin xyz='{L1} 0 0' rpy='0 0 0'/>
    <axis xyz='0 0 1'/>
    <dynamics damping='0.05' friction='0.025'/>
    <limit lower='-{math.pi}' upper='{math.pi}' effort='35' velocity='8'/>
  </joint>

  <link name='link2'>
    <inertial><mass value='{m2}'/>
      <inertia ixx='{ixx2}' iyy='{iyy2}' izz='{izz2}' ixy='0' ixz='0' iyz='0'/>
    </inertial>
    <visual>
      <origin xyz='{L2/2} 0 0.03' rpy='0 0 0'/>
      <geometry><box size='{L2} {w} {h}'/></geometry>
    </visual>
    <collision>
      <origin xyz='{L2/2} 0 0.03' rpy='0 0 0'/>
      <geometry><box size='{L2} {w} {h}'/></geometry>
    </collision>
  </link>

  <joint name='ee_fixed' type='fixed'>
    <parent link='link2'/>
    <child link='ee'/>
    <origin xyz='{L2} 0 0' rpy='0 0 0'/>
  </joint>

  <link name='ee'>
    <inertial><mass value='{mee}'/>
      <inertia ixx='{ixxe}' iyy='{iyye}' izz='{izze}' ixy='0' ixz='0' iyz='0'/>
    </inertial>
    <visual>
      <origin xyz='0 0 0.04' rpy='0 0 0'/>
      <geometry><cylinder radius='0.02' length='0.01'/></geometry>
    </visual>
  </link>
</robot>
"""


# ---------------------------------
# Bullet connect & step
# ---------------------------------

def connect(gui=False):
    cid = p.connect(p.GUI if gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setTimeStep(1/240.0)
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    return cid


def step(gui=False):
    p.stepSimulation()
    if gui:
        time.sleep(1/240.0)


# ---------------------------------
# 2R Analytic IK on XY plane
# ---------------------------------

def ik2r_xy(x, y, L1, L2, elbow_up=True):
    r = math.hypot(x, y)
    r = clamp(r, 1e-6, L1 + L2 - 1e-6)
    c2 = (r*r - L1*L1 - L2*L2)/(2*L1*L2)
    c2 = clamp(c2, -1.0, 1.0)
    s2 = (1.0 if elbow_up else -1.0) * math.sqrt(max(0.0, 1.0 - c2*c2))
    th2 = math.atan2(s2, c2)
    k1, k2 = L1 + L2*c2, L2*s2
    th1 = math.atan2(y, x) - math.atan2(k2, k1)
    # wrap
    th1 = (th1 + math.pi)%(2*math.pi) - math.pi
    th2 = (th2 + math.pi)%(2*math.pi) - math.pi
    return th1, th2


# ---------------------------------
# Main
# ---------------------------------

def run(traj: str = "circle", gui: bool = False):
    L1, L2 = 0.25, 0.20
    dt = 1/240.0

    connect(gui)
    p.resetDebugVisualizerCamera(1.25, 45, -30, [0.25, 0.0, 0.03])

    arm = p.loadURDF(write_tmp(two_link_urdf(L1, L2), "two_link_min.urdf"),
                     basePosition=[0,0,0], useFixedBase=True)
    ee_link = p.getNumJoints(arm) - 1  # link index of 'ee'

    # Grid & workspace rings
    for i in range(-5, 11):
        p.addUserDebugLine([i*0.05, -0.5, 0], [i*0.05, 0.5, 0], [0.7,0.7,0.7], 0)
    for j in range(-10, 11):
        p.addUserDebugLine([-0.25, j*0.05, 0], [0.75, j*0.05, 0], [0.7,0.7,0.7], 0)

    Rmax, Rmin = L1+L2, abs(L1-L2)
    Nring = 64
    for R, col in [(Rmax,[0.6,0.8,0.9]), (Rmin,[0.9,0.7,0.6])]:
        for k in range(Nring):
            a0=2*math.pi*k/Nring; a1=2*math.pi*((k+1)%Nring)/Nring
            p.addUserDebugLine([R*math.cos(a0), R*math.sin(a0), 0.001],
                               [R*math.cos(a1), R*math.sin(a1), 0.001], col, 0)

    # State
    elbow_up = True
    paused = False
    speed = 1.0

    manual = False
    manual_ang = 0.0
    manual_r = L1
    manual_target = L1
    radial_speed = 0.45  # m/s

    # Smooth handover
    handover = False
    handover_alpha = 0.0
    handover_dur = 0.6
    hold_pos = np.array([L1, 0.0, 0.03], dtype=float)

    # Trace throttling
    trace = deque(maxlen=600)
    trace_tick = 0
    TRACE_EVERY_N = 6
    MIN_SEG_LEN   = 0.002
    TRACE_LIFE    = 0.8

    # Target marker throttling
    tgt_id = None
    tgt_tick = 0
    TGT_EVERY_N = 8
    TGT_LIFE    = 0.5

    center = np.array([0.30, 0.00, 0.03])
    t = 0.0

    def default_target(tt):
        if traj == 'lem':
            a=0.18
            return np.array([center[0] + a*math.sin(0.9*tt),
                             center[1] + a*math.sin(1.8*tt), 0.03])
        r=0.16
        return np.array([center[0] + r*math.cos(0.6*tt),
                         center[1] + r*math.sin(0.6*tt), 0.03])

    print("[Keys] F/B extend-retract, S smooth handover, E elbow, +/- speed, P pause, R reset, C clear")

    while True:
        if gui:
            keys = p.getKeyboardEvents()
            # speed
            if key_edge(keys, *K_PLUS):
                speed = min(3.0, speed*1.2)
                print('[key] + : speed ->', round(speed,4))
            if key_edge(keys, *K_MINUS):
                speed = max(0.2, speed/1.2)
                print('[key] - : speed ->', round(speed,4))
            # elbow
            if key_edge(keys, *K_E):
                elbow_up = not elbow_up
                print('[key] E : elbow_up =', elbow_up)
            # pause
            if key_edge(keys, *K_P):
                paused = not paused
                print('[key] P : paused =', paused)
            # reset
            if key_edge(keys, *K_R):
                p.resetJointState(arm,0,0); p.resetJointState(arm,1,0)
                manual=False; handover=False
                trace.clear(); t=0.0
                print('[key] R : reset')
            if key_edge(keys, *K_C):
                trace.clear(); print('[key] C : clear trace')
            # manual extend
            if key_edge(keys, *K_F):
                handover = False
                st = p.getLinkState(arm, ee_link, computeForwardKinematics=True)
                ee = np.array(st[4]) if st else np.array([L1,0,0])
                r_now = float(math.hypot(ee[0], ee[1]))
                js = p.getJointStates(arm,[0,1])
                q1 = js[0][0]
                ang = float(math.atan2(ee[1], ee[0])) if r_now>1e-6 else float(q1)
                manual_ang = ang
                manual_r = r_now
                manual_target = Rmax - 1e-3
                manual=True
                print('[key] F : EXTEND start; r=%.3f -> %.3f, ang=%.2f°' % (r_now, manual_target, math.degrees(ang)))
            # manual retract
            if key_edge(keys, *K_B):
                handover = False
                st = p.getLinkState(arm, ee_link, computeForwardKinematics=True)
                ee = np.array(st[4]) if st else np.array([L1,0,0])
                r_now = float(math.hypot(ee[0], ee[1]))
                js = p.getJointStates(arm,[0,1])
                q1 = js[0][0]
                ang = float(math.atan2(ee[1], ee[0])) if r_now>1e-6 else float(q1)
                manual_ang = ang
                manual_r = r_now
                manual_target = max(Rmin + 1e-3, 0.02)
                manual=True
                print('[key] B : RETRACT start; r=%.3f -> %.3f, ang=%.2f°' % (r_now, manual_target, math.degrees(ang)))
            # smooth handover
            if key_edge(keys, *K_S):
                try:
                    hold_pos = np.array([manual_r*math.cos(manual_ang), manual_r*math.sin(manual_ang), 0.03], dtype=float)
                except Exception:
                    st = p.getLinkState(arm, ee_link, computeForwardKinematics=True)
                    ee = np.array(st[4]) if st else np.array([L1,0,0])
                    hold_pos = np.array([ee[0], ee[1], 0.03], dtype=float)
                manual = False
                handover = True
                handover_alpha = 0.0
                print('[key] S : handover -> auto start')

        if not paused:
            # choose desired EE target
            if manual:
                dr = clamp(manual_target - manual_r, -radial_speed*dt, radial_speed*dt)
                manual_r = clamp(manual_r + dr, Rmin+1e-3, Rmax-1e-3)
                des = np.array([manual_r*math.cos(manual_ang), manual_r*math.sin(manual_ang), 0.03])
                if abs(manual_target - manual_r) < 1e-3:
                    manual_target = manual_r  # hold
            elif handover:
                default_des = default_target(t)
                handover_alpha = min(1.0, handover_alpha + dt/ max(1e-6, handover_dur))
                des = (1.0 - handover_alpha) * hold_pos + handover_alpha * default_des
                if handover_alpha >= 1.0:
                    handover = False
            else:
                des = default_target(t)

            # target marker (throttled)
            tgt_tick += 1
            if tgt_tick % TGT_EVERY_N == 0:
                if tgt_id is not None:
                    p.removeUserDebugItem(tgt_id)
                tgt_id = p.addUserDebugLine(des, des+np.array([0,0,0.03]), [0,1,0], 2, TGT_LIFE)

            # IK -> desired q
            th1, th2 = ik2r_xy(float(des[0]), float(des[1]), L1, L2, elbow_up)

            # POSITION_CONTROL with capped velocity
            p.setJointMotorControl2(arm, 0, p.POSITION_CONTROL, targetPosition=th1,
                                    positionGain=0.4, velocityGain=1.0, force=40.0, maxVelocity=6.0*speed)
            p.setJointMotorControl2(arm, 1, p.POSITION_CONTROL, targetPosition=th2,
                                    positionGain=0.4, velocityGain=1.0, force=35.0, maxVelocity=6.0*speed)

            # EE trace (throttled)
            st = p.getLinkState(arm, ee_link, computeForwardKinematics=True)
            if st is not None:
                ee = np.array(st[4])
                trace_tick += 1
                if trace_tick % TRACE_EVERY_N == 0:
                    if len(trace)==0 or np.linalg.norm(ee - trace[-1]) >= MIN_SEG_LEN:
                        trace.append(ee)
                        if len(trace) >= 2:
                            p.addUserDebugLine(trace[-2], trace[-1], [0.1,0.9,0.1], TRACE_LIFE)

            t += dt

        step(gui)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--traj", choices=["circle","lem"], default="circle")
    args = ap.parse_args()
    run(traj=args.traj, gui=args.gui)
