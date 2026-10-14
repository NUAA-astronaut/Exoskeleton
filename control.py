"""Controller receives delayed IMUs and optional motor encoders only."""
from collections import deque
import numpy as np

class ImuPair:
    def __init__(self,c):
        self.c=c; self.rng=np.random.default_rng(c['seed']); self.queue=deque()
        self.latest=None; self.next_sample=0
        self.bias=self.rng.normal(0,np.deg2rad(c['pitch_bias_deg']),2)

    def read(self,m,d,t):
        if t+1e-9 >= self.next_sample:
            packets=[]
            for i,side in enumerate(('left','right')):
                acc=d.sensor('acc_'+side).data.copy()
                gyro=d.sensor('gyro_'+side).data.copy()
                quat=d.sensor('quat_'+side).data.copy() # MuJoCo w,x,y,z
                w,x,y,z=quat
                pitch=-np.arcsin(np.clip(2*(w*y-z*x),-1,1))
                pitch+=self.bias[i]+self.rng.normal(0,np.deg2rad(self.c['pitch_noise_deg']))
                acc=np.clip(acc+self.rng.normal(0,self.c['accel_noise_m_s2'],3),-6*9.81,6*9.81)
                gyro=np.clip(gyro+self.rng.normal(0,self.c['gyro_noise_rad_s'],3),-np.deg2rad(2000),np.deg2rad(2000))
                packets.append(dict(pitch=pitch,velocity=-gyro[1],acc=acc,gyro=gyro))
            self.queue.append((t+self.c['imu_delay_s'],packets))
            self.next_sample+=1/self.c['imu_hz']
        while self.queue and self.queue[0][0]<=t+1e-9:
            self.latest=self.queue.popleft()[1]
        return self.latest

class Assist:
    def __init__(self,c,mode,enabled=True):
        self.c=c; self.mode=mode; self.enabled=enabled
        self.history=[deque(maxlen=int(2*c['cycle_s']/c['dt_s'])) for _ in range(2)]
        self.torque=np.zeros(2); self.phase=np.zeros(2)

    @staticmethod
    def pulse(p,start,end):
        return np.sin(np.pi*(p-start)/(end-start))**2 if start<p<end else 0.

    def update(self,packets,encoder_speed):
        target=np.zeros(2)
        if packets is not None:
            for i,packet in enumerate(packets):
                angle,vel=packet['pitch'],packet['velocity']
                self.history[i].append(angle)
                h=self.history[i]
                if len(h)<int(self.c['cycle_s']/self.c['dt_s']):
                    continue
                center=(max(h)+min(h))/2
                self.phase[i]=(np.arctan2(-vel*self.c['cycle_s']/(2*np.pi),angle-center)/(2*np.pi))%1
                p=(self.phase[i]+self.c['phase_shift_cycles'])%1
                if max(h)-min(h)<np.deg2rad(8):
                    continue
                amp=min(self.c['assist_nm'],self.c['motor_rated_nm'])
                # Phase zero is approximate maximum thigh flexion, not heel strike.
                if self.mode=='downstairs':
                    target[i]=-min(amp,1.5)*np.tanh(encoder_speed[i]/.5)*self.pulse(p,.02,.45)
                else:
                    target[i]=amp*(-self.pulse(p,.02,.38)+(.65 if self.mode=='upstairs' else .8)*self.pulse(p,.55,.95))
        if not self.enabled:
            target[:]=0
        dt=self.c['dt_s']
        delta=(target-self.torque)*(1-np.exp(-dt/self.c['motor_lag_s_assumed']))
        self.torque+=np.clip(delta,-self.c['motor_slew_nm_s_assumed']*dt,self.c['motor_slew_nm_s_assumed']*dt)
        return np.clip(self.torque,-self.c['motor_rated_nm'],self.c['motor_rated_nm']).copy()
