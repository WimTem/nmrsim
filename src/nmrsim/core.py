import SLEEPY as sl
import numpy as np
import matplotlib.pyplot as plt

# Experimental conditions

v0H = 600          # 1H frequency in MHz
vr = 20e3          # MAS frequency in Hz

# Powder averaging
pwdavg = 2
n_gamma = 10

# CP
v1H = 60e3         # 1H RF field, Hz
v1C = 60e3         # 13C RF field, Hz
tCP = 0.5e-3       # CP contact time, 0.5 ms

# Direct (13C) acquisition
SW1 = 100e3        # indirect dimension bandwidth, Hz
SW2 = 100e3        # direct dimension bandwidth, Hz

n1 = 64             # t1 points
n2 = 256            # t2 points

dt2 = 1 / SW2

# CH bond length
rCH = .109         # nm = 1.09 Angstrom

print("t2 dwell =", dt2 * 1e6, "us")
print("t2 acquisition =", n2 * dt2 * 1e3, "ms")

ex = sl.ExpSys(
    v0H=v0H,
    Nucs=['1H', '13C'],
    vr=vr,
    pwdavg=pwdavg,
    n_gamma=n_gamma
)

# CH dipolar coupling
delta_CH = sl.Tools.dipole_coupling(rCH,'1H','13C')

print(f"CH dipolar coupling = {delta_CH/1e3:.1f} kHz")

ex.set_inter('dipole',i0=0,i1=1,delta=delta_CH)

# Add simple chemical shifts
ex.set_inter('CS', i=0, ppm=20.0)
ex.set_inter('CS', i=1, ppm=40.0)

L = ex.Liouvillian()


# CP pulse sequence
seqCP = L.Sequence()
seqCP.add_channel('13C', v1=v1C)
seqCP.add_channel('1H', v1=v1H)

# CP transfer test
rho = sl.Rho(rho0='1Hx', detect=['1Hx', '13Cx'])
rho()

seqCP * rho

rho()

print('Yes')

print("1H initial :", np.round(rho.I[0,0].real))
print("1H after CP:", np.round(rho.I[0,1].real))

print("13C initial :", np.round(rho.I[1,0].real))

print("13C after CP:", np.round(rho.I[1,1].real))

# CP contact-time optimization
contact_times = np.linspace(0.02e-3, 2.0e-3, 40)

C13 = []

for tCP_test in contact_times:

    seq = L.Sequence(Dt=tCP_test)

    seq.add_channel('13C', v1=v1C)
    seq.add_channel('1H', v1=v1H)
    rho_test = sl.Rho(rho0='1Hx', detect='13Cx')
    rho_test()
    seq*rho_test
    rho_test()

    C13.append(rho_test.I[0,1].real)

plt.figure(figsize=(7,4))

plt.plot(
    contact_times * 1e3,
    C13,
    'o-'
)

plt.xlabel('CP contact time (ms)')
plt.ylabel(r'$^{13}$C magnetization')
plt.title(r'$^1$H $\rightarrow$ $^{13}$C CP transfer')

plt.grid(alpha=0.2)
plt.show()