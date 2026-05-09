import numpy as np
from scipy.io import loadmat
from scipy.interpolate import CubicSpline

mat = loadmat('data/8. IGBT Accelerated Aging/IGBTAgingData_04022009/Data/Thermal Overstress Aging with Square Signal at gate and SMU data/Aging Data/Device 3/Device3  1.mat')
trans = mat['measurement'][0,0]['transient'].reshape(-1)[0]
v_gate = trans['timeDomain'].reshape(-1)[0]['gateSignalVoltage'].flatten()
v_ge = trans['timeDomain'].reshape(-1)[0]['gateEmitterVoltage'].flatten()
ic = trans['timeDomain'].reshape(-1)[0]['collectorEmitterCurrentSignal'].flatten()

gate_diff = np.diff(v_gate)
rising_idx = np.argmax(gate_diff)
start_idx = max(0, rising_idx - 10)
end_idx = min(len(v_ge), rising_idx + 100)

x = np.arange(end_idx - start_idx)
vge_window = v_ge[start_idx:end_idx]
ic_window = ic[start_idx:end_idx]

cs_vge = CubicSpline(x, vge_window)
cs_ic = CubicSpline(x, ic_window)
x_dense = np.linspace(0, len(x) - 1, len(x) * 50)
vge_dense = cs_vge(x_dense)
ic_dense = cs_ic(x_dense)

cross_idx = np.where(ic_dense >= 0.25)[0]
vt = np.nan
for idx in cross_idx:
    if idx < 15 or idx >= len(ic_dense) - 15:
        continue
    window_diffs = np.diff(ic_dense[idx-10:idx+10])
    if np.all(window_diffs >= -1e-4) and (ic_dense[idx+5] > ic_dense[idx-5]):
        vt = vge_dense[idx]
        break

print("Extracted Vth:", vt)
