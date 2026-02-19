
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from sympy import sqrt as sympy_sqrt

# Time array for plotting
time_receiver = np.arange(0, 241, 1)

# Default doppler shift parameters
f0 = 150
tprime0 = 120
c = 320
v0 = 80
L = 4000
org = True

#Get Symbolic Doppler Expresion to take Derivatives of
sf0, sv0, stprime, stprime0, sl, sc = sp.symbols('f0 v0 tprime tprime0 l c')
if org:
    t = stprime - stprime0
else:
    t = stprime - stprime0 + sl/sc
f = sf0/(1+(sv0/sc)*(sv0* (t- sympy_sqrt(t**2-(1-sv0**2/sc**2)*(t**2-sl**2/sc**2)))/(1-sv0**2/sc**2))/(sympy_sqrt(sl**2+(sv0* (t- sympy_sqrt(t**2-(1-sv0**2/sc**2)*(t**2-sl**2/sc**2)))/(1-sv0**2/sc**2))**2)))


def get_t(v0, L, c, tprime0, time_receiver, org=False):
    """Calculate time in aircraft reference frame from 
    spectrogram reference frame.

    :type v0: numpy.int64 float
    :param v0: Velocity of the aircraft
    :type L: numpy.int64 float
    :param L: Altitude of the aircraft
    :type c: numpy.int64 float
    :param c: Velocity of the wave propagation
    :type tprime0: numpy.int64 float
    :param tprime0: Time of closest approach in spectrogram reference frame
    :type time_receiver: np.array float
    :param time_receiver: Time array in spectrogram reference frame
    :rtype: np.array
    :return: Time array in aircraft reference frame
    """
    beta = v0/c
    if org:
        arg = time_receiver - tprime0
    else:
        arg = time_receiver - tprime0 + L/c
    discriminant = arg**2 - (1 - beta**2) * (arg**2 - (L/c)**2)

    return (arg - np.sqrt(discriminant)) / (1 - beta**2)


def get_f(v0, L, c, t_array, f0):
    """Calculate observed frequency (Doppler effect).

    :type v0: numpy.int64 float
    :param v0: Velocity of the aircraft
    :type L: numpy.int64 float
    :param L: Altitude of the aircraft
    :type c: numpy.int64 float
    :param c: Velocity of the wave propagation
    :type t_array: np.array
    :param t_array: Time array in aircraft reference frame
    :type f0: numpy.int64 float
    :param f0: Emitted frequency from aircraft
    :rtype: np.array
    :return: Frequency array of received frequency at seonsors corresponding 
    to t_array
    """
    base = np.sqrt(L**2 + (v0 * t_array)**2)
    base_checked = np.where(base != 0, base, 1e-10)
    return f0 / (1 + (v0/c) * (v0*t_array) / base_checked)

#take derivative with respect to c
for var in [sf0, sv0, stprime, stprime0, sl, sc]:
    df = sp.diff(f, var)
    print("Derivatie with respect to {}: {}".format(var, df))


plt.figure(figsize=(10, 6))

tprime = get_t(v0,L,c,tprime0,time_receiver)
ft = get_f(v0,L,c,tprime, f0)

plt.plot(time_receiver, ft, 'blue', linewidth=0.5, zorder=10)
plt.axvline(tprime0, c='blue', linewidth=0.5, zorder=10)


tprime = get_t(v0,L,c,tprime0,time_receiver,org=True)
ft = get_f(v0,L,c,tprime, f0)

plt.plot(time_receiver, ft, 'red', linewidth=0.5, zorder=10)
plt.axvline(tprime0, c ='red', linewidth=0.5, zorder=10)

plt.show()


