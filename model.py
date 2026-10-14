"""Parametric sagittal human + ideal coaxial bilateral hip exoskeleton."""
import math
from pathlib import Path
import numpy as np

MODES = ('level', 'upstairs', 'downstairs')

def surface(x, mode, c):
    if mode == 'level':
        return 0.0
    return (1 if mode == 'upstairs' else -1) * math.floor((x + c['step_depth_m']/2)/c['step_depth_m']) * c['step_height_m']

def build(mode, c, destination):
    L = c['height_m'] * .245
    mass = c['human_mass_kg']
    ground = '<geom name="floor" type="plane" size="0 0 .1" rgba=".22 .26 .30 1"/>' if mode == 'level' else ''
    if mode != 'level':
        for k in range(-4, 30):
            x = k*c['step_depth_m']; top = surface(x, mode, c)
            bottom = -6
            ground += f'<geom name="step_{k+4}" type="box" pos="{x} 0 {(top+bottom)/2}" size="{c["step_depth_m"]/2} .7 {(top-bottom)/2}" rgba=".35 .40 .46 1"/>'
    legs, sensors, actuators = '', '', ''
    for side, y in [('left', .10), ('right', -.10)]:
        legs += f'''
        <geom name="motor_case_{side}" type="cylinder" pos="0 {y*1.7} 0" euler="90 0 0" size=".045 .025" mass="{c['motor_mass_kg_assumed']}" rgba=".16 .18 .20 1" contype="0" conaffinity="0"/>
        <body name="thigh_{side}" pos="0 {y} 0">
          <joint name="hip_{side}" axis="0 -1 0" range="-45 120"/>
          <geom type="capsule" fromto="0 0 0 0 0 {-L}" size=".055" mass="{mass*.10}"/>
          <geom type="capsule" fromto="0 {y*.4} -.02 0 {y*.4} {-L*.65}" size=".012" mass=".15" rgba=".1 .6 .75 1"/>
          <site name="imu_{side}" pos=".065 0 {-L*.48}" size=".025 .018 .008" type="box" rgba="1 .55 .08 1"/>
          <body name="shank_{side}" pos="0 0 {-L}">
            <joint name="knee_{side}" axis="0 1 0" range="0 150"/>
            <geom type="capsule" fromto="0 0 0 0 0 {-L}" size=".04" mass="{mass*.0465}"/>
            <body name="foot_{side}" pos="0 0 {-L}">
              <joint name="ankle_{side}" axis="0 -1 0" range="-70 70"/>
              <geom name="sole_{side}" type="box" pos=".025 0 -.025" size=".12 .045 .025" mass="{mass*.0145}" contype="1" conaffinity="1" rgba=".15 .20 .28 1"/>
            </body>
          </body>
        </body>'''
        sensors += f'<accelerometer name="acc_{side}" site="imu_{side}"/><gyro name="gyro_{side}" site="imu_{side}"/><framequat name="quat_{side}" objtype="site" objname="imu_{side}"/><magnetometer name="mag_{side}" site="imu_{side}"/>'
        actuators += f'<motor name="exo_{side}" joint="hip_{side}" gear="1" ctrllimited="true" ctrlrange="-{c["motor_peak_nm"]} {c["motor_peak_nm"]}"/>'
    xml = f'''<mujoco model="hip_exo_{mode}">
    <compiler angle="degree"/>
    <option timestep="{c['dt_s']}" integrator="implicitfast" gravity="0 0 -9.81"/>
    <visual><global offwidth="1200" offheight="800"/></visual>
    <default><joint damping="1" armature=".025" limited="true"/><geom friction=".9 .01 .001" contype="0" conaffinity="0" rgba=".75 .65 .53 1"/></default>
    <worldbody><light pos="0 -3 5" dir="0 0 -1"/>
    <geom type="plane" pos="0 0 -6.1" size="0 0 .1" rgba=".15 .18 .22 1"/>
    <body name="terrain">{ground}</body>
    <body name="pelvis" pos="0 0 0">
      <joint name="base_x" type="slide" axis="1 0 0" limited="false" damping="0" armature="0"/>
      <joint name="base_z" type="slide" axis="0 0 1" limited="false" damping="0" armature="0"/>
      <joint name="base_pitch" axis="0 -1 0" range="-45 45"/>
      <geom type="box" size=".10 .14 .075" mass="{mass*.15}"/>
      <geom type="capsule" fromto="0 0 .10 0 0 .52" size=".12" mass="{mass*.528}"/>
      {legs}
    </body></worldbody><actuator>{actuators}</actuator><sensor>{sensors}</sensor></mujoco>'''
    # Only feet collide with terrain; limb/torso collisions intentionally excluded.
    xml = xml.replace('<body name="terrain">', '<body name="terrain"><!-- contact terrain -->')
    import xml.etree.ElementTree as ET
    root = ET.fromstring(xml)
    for geom in root.find('worldbody').find('body').findall('geom'):
        geom.set('contype', '1'); geom.set('conaffinity', '1')
    Path(destination).write_text(ET.tostring(root, encoding='unicode'), encoding='utf-8')

def reference(t, mode, c):
    """Synthetic alternating stance/swing IK, not measured human gait."""
    L = c['height_m']*.245
    feet = []
    for offset, index in [(0, 0), (.5, -1)]:
        s = t/c['cycle_s']+offset
        n = math.floor(s); p = s-n
        x0 = (2*n+index)*c['step_depth_m']; x1 = x0+2*c['step_depth_m']
        z0, z1 = surface(x0, mode, c), surface(x1, mode, c)
        u = np.clip((p-.60)/.40, 0, 1)
        horizontal = np.clip((u-.20)/.6,0,1)
        w = horizontal**3*(10+horizontal*(-15+6*horizontal))
        # Lift early before crossing the next stair riser, then lower late.
        lift_phase=min(u/.20,(1-u)/.20,1)
        lift = (.06 + (c['step_height_m'] if mode == 'upstairs' else 0))*lift_phase**3*(10+lift_phase*(-15+6*lift_phase))
        wz=w
        if mode=='downstairs':
            lower=np.clip((u-.8)/.2,0,1)
            wz=lower**3*(10+lower*(-15+6*lower))
        feet.append((x0+(x1-x0)*w, z0+(z1-z0)*wz+.052+lift))
    bx = (feet[0][0]+feet[1][0])/2
    # Smooth minimum keeps the lower leg reachable during stair swing.
    bz = (feet[0][1]+feet[1][1]-math.sqrt((feet[0][1]-feet[1][1])**2+.03**2))/2 + L*1.78
    q = [bx, bz, 0.0]
    for x,z in feet:
        dx,dz = x-bx,z-bz
        r = math.hypot(dx,dz)
        if r >= 2*L:
            raise ValueError('Unreachable gait: adjust height, stair dimensions or pelvis height')
        a = math.acos(r/(2*L))
        hip = math.atan2(dx,-dz)+a
        knee = 2*a
        q.extend([hip,knee,knee-hip])
    return np.array(q)

def reference_state(t, mode, c):
    h = 1e-4
    q = reference(t,mode,c)
    qm,qp = reference(t-h,mode,c),reference(t+h,mode,c)
    return q,(qp-qm)/(2*h),(qp-2*q+qm)/h**2


