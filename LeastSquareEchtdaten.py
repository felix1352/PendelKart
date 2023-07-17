from load_pendel_data import LoadHdf5Mat
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
from dynamics import get_dxdt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

plt.close('all')
dx_dt = get_dxdt()

#Gesamte Messdaten einlesen
path = Path('joystick_control.mat')
data_upswing = LoadHdf5Mat(path)
time = data_upswing.get('X').get('Data')
data_names = data_upswing.get('Y').get('Name')
force = data_upswing.get('Y').get('Data')[0]
pos = data_upswing.get('Y').get('Data')[1]
angle = data_upswing.get('Y').get('Data')[3]

dt = time[1]-time[0]
#N = 80 #Werte für Least Squares
#Ersten N Messdaten auslesen
time_leastsquares = time[3000:9000]
angle_leastsquares = np.array(angle[3000:9000]) - 180
pos_leastsquares = pos[3000:9000]
force_leastsquares = force[3000:9000]
ytrain = np.concatenate((angle_leastsquares, pos_leastsquares))

#Least Square Parameter Estimation
x0 = [-3780.08789, 0, 0.1027, 0]
def system_gleichungen(params, t, x0):
    parameter = {
        'g': 9.81,
        'M': params[0],
        'm': params[1],
        'l': params[2],
        'b': params[3],
    }
    t_start = time_leastsquares[0]
    t_end = time_leastsquares[-1]
    resest = solve_ivp(lambda t, x: dx_dt(t, x, parameter), [t_start, t_end], x0, t_eval = time_leastsquares)
    return [resest.y[0], resest.y[2]]

def residuals(params, t, y):
    exp = system_gleichungen(params, t, x0)
    res1 = angle_leastsquares - exp[0]
    res2 = pos_leastsquares - exp[1]
    res1 = np.squeeze(res1)
    res2 = np.squeeze(res2)
    return np.concatenate((res1, res2))

paramsinitial = [1, 1, 1, 1]

result = least_squares(residuals, paramsinitial, args=(time_leastsquares, ytrain ))
params_est = result.x
print('Die geschätzten Werte sind: ', params_est)

ergebnis = system_gleichungen(params_est, 0, x0)
plt.subplot(1,2,1)
plt.plot(time_leastsquares, ergebnis[0], label='prediction angle')
plt.plot(time_leastsquares,  angle_leastsquares, label='measured angle')
plt.legend()

plt.subplot(1,2,2)
plt.plot(time_leastsquares, ergebnis[1], label='prediction position')
plt.plot(time_leastsquares, pos_leastsquares, label='measured position')
plt.legend()
plt.show()


