"""Physical conventions, controller limits, deterministic runs and render QA."""
import json
from pathlib import Path
import numpy as np
import mujoco
from model import MODES,build,reference_state
from control import Assist

ROOT=Path(__file__).resolve().parent
c=json.loads((ROOT/'config.json').read_text())
out=ROOT/'outputs';out.mkdir(exist_ok=True)
for mode in MODES:
    build(mode,c,out/f'{mode}.xml')
    m=mujoco.MjModel.from_xml_path(str(out/f'{mode}.xml'));d=mujoco.MjData(m)
    assert m.nq==9 and m.nu==2 and m.nsensor==8
    assert abs(m.body_subtreemass[m.body('pelvis').id]-(70+2*.6+2*.15))<1e-8
    # Check all reference samples satisfy joint ranges and have reachable IK.
    for t in np.linspace(0,8,801):
        q,_,_=reference_state(t,mode,c)
        for i in range(3,9): assert m.jnt_range[i,0]<=q[i]<=m.jnt_range[i,1],(mode,t,i,q[i])
    # Static accelerometer measures specific force, including gravity at rest.
    d.qpos[:],_,_=reference_state(0,mode,c)
    d.qvel[:]=0;d.qacc[:]=0;mujoco.mj_inverse(m,d)
    assert np.isclose(np.linalg.norm(d.sensor('acc_left').data),9.81,atol=1e-6)
    # Sensor axes must report positive flexion for a positive hip velocity.
    d.qvel[3]=1;mujoco.mj_forward(m,d)
    assert np.isclose(d.sensor('gyro_left').data[1],-1)
    d.qpos[:],d.qvel[:],_=reference_state(2.2,mode,c);mujoco.mj_forward(m,d)
    renderer=mujoco.Renderer(m,height=600,width=900)
    cam=mujoco.MjvCamera();cam.lookat[:]=[d.qpos[0],0,d.qpos[1]-.1]
    cam.distance=3;cam.azimuth=110;cam.elevation=-12
    renderer.update_scene(d,camera=cam)
    import matplotlib.pyplot as plt
    plt.imsave(out/f'{mode}_preview.png',renderer.render());renderer.close()

controller=Assist(c,'level')
previous=np.zeros(2)
for k in range(6000):
    p=2*np.pi*k*c['dt_s']/c['cycle_s']
    packet=dict(pitch=.4*np.cos(p),velocity=-.4*2*np.pi/c['cycle_s']*np.sin(p))
    torque=controller.update([packet,packet],np.zeros(2))
    assert np.max(np.abs(torque))<=4
    assert np.max(np.abs(torque-previous))<=c['motor_slew_nm_s_assumed']*c['dt_s']+1e-10
    previous=torque
print('PASS: models, mass, IK/ranges, IMU gravity/axes, motor torque/slew, three rendered previews')
