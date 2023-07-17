import numpy as np
from dynamics import get_dxdt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

dx_dt = get_dxdt()

dt = 0.01
t_start = 0.0
t_end = 5
x0 = [0, 0, 0, 0]  # Anfangswerte für phi, dphi, s, ds

t_eval = np.arange(t_start, t_end+dt, dt) #Vektor der für gleichmäßige Zeitabstände beim Lösen sorgt, später wichtig wegen realer Abtastung
res = solve_ivp(dx_dt,[t_start,t_end],x0, t_eval = t_eval) #voltage in dynamics definiert

t_train = res.t
noise1 = np.random.normal(0,0.05,len(t_train))
noise2 = np.random.normal(0,0.1,len(t_train))

angle_plot = res.y[0]
pos_plot = res.y[2]

angle_train = res.y[0] + noise1
pos_train = res.y[2] + noise2
y_train = np.concatenate((angle_train, pos_train)) #aneinanderhängen der trainingsdaten damit man später die Differenz zweier arrays minimieren kann

#Plotten

plt.subplot(2,1,1)
plt.plot(t_train, angle_train, label='simulated angle')
plt.xlabel('Zeit in s')
plt.ylabel('phi in rad')

plt.subplot(2,1,2)
plt.plot(t_train, pos_train, label='simulated position')
plt.xlabel('Zeit in s')
plt.ylabel('s in m')


#Least Square Parameter Estimation
paramsecht = [0.7, 0.221, 0.5, 0.1]
print('Die echten Werte sind: ', paramsecht)
def system_gleichungen(params, t, x):
    parameter = {
        'g': 9.81,
        'M': params[0],
        'm': params[1],
        'l': params[2],
        'b': params[3],
    }
    resest = solve_ivp(lambda t, x: dx_dt(t, x, parameter), [t_start, t_end], x0, t_eval = t_eval)
    return [resest.y[0], resest.y[2]]

def residuals(params, t, x):
    exp = system_gleichungen(params, t, x)
    res1 = angle_train- exp[0]
    res2 = pos_train - exp[1]
    return np.concatenate((res1, res2))

paramsinitial = [1, 1, 1, 1]

result = least_squares(residuals, paramsinitial, args=(t_train, y_train))
params_est = result.x
print('Die geschätzten Werte sind: ', params_est)
f = abs(params_est-paramsecht)
print('Der betragsmäßige Fehler ist: ', f)

#Plotten zum Vergleich
erg_plot = system_gleichungen(params_est, 0, 0)
plt.subplot(2, 1, 1)
plt.plot(t_train, erg_plot[0], '--',  label = 'predicted angle')
plt.legend()
plt.title('Methode der kleinsten Quadrate am Einfachpendel mit Wagen in der Simulation')

plt.subplot(2, 1, 2)
plt.plot(t_train, erg_plot[1], '--', label = 'predicted position')
plt.legend()
plt.show()

correlation, _ = pearsonr(angle_plot, erg_plot[0])
print('Korrelation Winkel: ',correlation)

correlation, _ = pearsonr(pos_plot, erg_plot[1])
print('Korrelation Position: ',correlation)

