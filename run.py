import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np
import mujoco
from model import MODES,build,reference_state
from control import ImuPair,Assist

ROOT=Path(__file__).resolve().parent

def simulate(mode,c,seconds,enabled,out,viewer=False):
    out.mkdir(parents=True,exist_ok=True)
    path=out/f'{mode}.xml'; build(mode,c,path)
    m=mujoco.MjModel.from_xml_path(str(path)); d=mujoco.MjData(m)
    inv=mujoco.MjData(m)
    d.qpos[:],d.qvel[:],_=reference_state(0,mode,c)
    mujoco.mj_forward(m,d)
    imu=ImuPair(c); controller=Assist(c,mode,enabled)
    rows=[]; hip=np.array([3,6]); dt=c['dt_s']
    display=None
    if viewer:
        from mujoco import viewer as mjviewer
        display=mjviewer.launch_passive(m,d)
        display.cam.azimuth=90; display.cam.elevation=-15; display.cam.distance=3.5
    try:
        for k in range(round(seconds/dt)):
            started=time.perf_counter(); t=k*dt
            q,v,a=reference_state(t,mode,c)
            packets=imu.read(m,d,t)
            tau=controller.update(packets,d.qvel[hip])
            # Virtual human inverse-dynamics tracker. All root generalized forces
            # are explicit artificial support, not exoskeleton assistance.
            inv.qpos[:]=d.qpos; inv.qvel[:]=d.qvel
            inv.qacc[:]=a+180*(q-d.qpos)+26*(v-d.qvel)
            mujoco.mj_inverse(m,inv)
            human=inv.qfrc_inverse.copy()
            d.qfrc_applied[:]=human
            d.ctrl[:]=tau
            if not np.all(np.isfinite(d.qpos)) or np.max(np.abs(d.qpos-q))>1:
                raise RuntimeError(f'Tracking failure at {t:.3f}s')
            if k%round(1/(c['imu_hz']*dt))==0:
                row=dict(time_s=t,tracking_error_rad_m=float(np.max(np.abs(d.qpos-q))),support_x_n=human[0],support_z_n=human[1],support_pitch_nm=human[2],contacts=d.ncon)
                for i,side in enumerate(('left','right')):
                    j=hip[i]
                    row.update({f'hip_{side}_rad':d.qpos[j],f'hip_speed_{side}_rad_s':d.qvel[j],f'human_{side}_nm':human[j],f'exo_{side}_nm':tau[i],f'exo_power_{side}_w':tau[i]*d.qvel[j],f'phase_{side}':controller.phase[i]})
                    if packets is None:
                        packet=dict(pitch=float('nan'),acc=np.full(3,np.nan),gyro=np.full(3,np.nan))
                    else:
                        packet=packets[i]
                    row[f'imu_pitch_{side}_rad']=packet['pitch']
                    for sensor in ('acc','gyro'):
                        for axis,value in zip('xyz',packet[sensor]): row[f'{sensor}_{side}_{axis}']=value
                rows.append(row)
            mujoco.mj_step(m,d)
            if display:
                if not display.is_running(): break
                if k%20==0:
                    display.cam.lookat[:]=[d.qpos[0],0,d.qpos[1]]; display.sync()
                time.sleep(max(0,dt-(time.perf_counter()-started)))
    finally:
        if display: display.close()
    tag=f'{mode}_{"assist" if enabled else "baseline"}'
    with (out/f'{tag}.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    summary={'mode':mode,'assistance':enabled,'samples':len(rows),'max_tracking_error_rad_m':max(r['tracking_error_rad_m'] for r in rows),'support_z_rms_n':float(np.sqrt(np.mean([r['support_z_n']**2 for r in rows])))}
    for side in ('left','right'):
        for source in ('human','exo'):
            values=np.array([r[f'{source}_{side}_nm'] for r in rows])
            summary[f'{source}_{side}_rms_nm']=float(np.sqrt(np.mean(values**2)))
        power=np.array([r[f'exo_power_{side}_w'] for r in rows])
        summary[f'exo_{side}_positive_j']=float(np.maximum(power,0).sum()/c['imu_hz'])
        summary[f'exo_{side}_negative_j']=float(np.minimum(power,0).sum()/c['imu_hz'])
    (out/f'{tag}_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (out/'config_used.json').write_text(json.dumps(c,indent=2),encoding='utf-8')
    return rows,summary

def plot(allrows,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(len(allrows),3,figsize=(15,3.5*len(allrows)),squeeze=False)
    for r,(mode,rows) in enumerate(allrows):
        t=[x['time_s'] for x in rows]
        for side in ('left','right'):
            for col,key in enumerate((f'imu_pitch_{side}_rad',f'exo_{side}_nm',f'exo_power_{side}_w')):
                axes[r,col].plot(t,[x[key] for x in rows],label=side,lw=1)
        for ax,title in zip(axes[r],('IMU pitch (rad)','Assist torque (Nm)','Mechanical power (W)')):
            ax.set_title(mode+' | '+title);ax.set_xlabel('Time (s)');ax.grid(alpha=.3);ax.legend()
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=150);plt.close(fig)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=(*MODES,'all'),default='all')
    p.add_argument('--seconds',type=float,default=8)
    p.add_argument('--viewer',action='store_true')
    p.add_argument('--baseline',action='store_true')
    p.add_argument('--compare',action='store_true')
    p.add_argument('--phase-shift',type=float,default=0)
    p.add_argument('--amplitude',type=float,default=3)
    p.add_argument('--out',type=Path,default=ROOT/'outputs')
    args=p.parse_args()
    c=json.loads((ROOT/'config.json').read_text())
    c['phase_shift_cycles']=args.phase_shift;c['assist_nm']=args.amplitude
    if not 0<args.seconds<=20: p.error('seconds must be in (0,20] for the finite staircase')
    if not 0<=args.amplitude<=c['motor_rated_nm']: p.error('amplitude must be between 0 and rated 4 Nm')
    allrows=[]
    for mode in MODES if args.mode=='all' else (args.mode,):
        rows,summary=simulate(mode,c,args.seconds,not args.baseline,args.out,args.viewer)
        print(json.dumps(summary));allrows.append((mode,rows))
        if args.compare: simulate(mode,c,args.seconds,False,args.out)
    plot(allrows,args.out)

if __name__=='__main__': main()
