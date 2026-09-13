import SLEEPY as sl
import numpy as np
import matplotlib.pyplot as pt

# Experimental conditions

v0H = 600          # 1H frequency in MHz
v0C = v0H * 0.25144953
vr = 20e3          # MAS frequency in Hz

# Powder averaging
pwdavg = 2
n_gamma = 10

# CP
v1H = 60e3         # 1H RF field, Hz
v1C = 60e3         # 13C RF field, Hz
tCP = 0.5e-3       # CP contact time, 0.5 ms

# Direct (13C) acquisition
SW1 = 12.5e3        # indirect dimension bandwidth, Hz
SW2 = 100e3        # direct dimension bandwidth, Hz

n1 = 64             # t1 points
n2 = 256            # t2 points

dt1 = 1/SW1
dt2 = 1/SW2

t1_axis = np.arange(n1) * dt1
t2_axis = np.arange(n2) * dt2

print("t1 maximum =", t1_axis[-1] * 1e3, "ms")
print("t2 maximum =", t2_axis[-1] * 1e3, "ms")

ex = sl.ExpSys(
    v0H=v0H,
    Nucs=['1H', '13C'],
    vr=vr,
    pwdavg=pwdavg,
    n_gamma=n_gamma
)

# CH dipolar coupling
rCH = 0.109 # nm = 1.09 Angstrom
delta_CH = sl.Tools.dipole_coupling(rCH,'1H','13C')

print(f"CH dipolar coupling = {delta_CH/1e3:.1f} kHz")
ex.set_inter('dipole',i0=0,i1=1,delta=delta_CH)

# Add simple chemical shifts
ex.set_inter('CS',i=0,ppm=2.0)
ex.set_inter('CS',i=1,ppm=40.0)

L = ex.Liouvillian()
seqAQ = L.Sequence(Dt=dt2)

def simulate_t1_point(L, t1, tCP, n2):
    # Start with proton transverse magnetization
    rho = sl.Rho(rho0='1Hp', detect='13Cp')

    # t1 evolution
    if t1 > 0:
        U1 = L.U(Dt=t1)
        U1 * rho
    # CP
    seqCP_local = L.Sequence(Dt=tCP)

    seqCP_local.add_channel('13C',v1=v1C)

    seqCP_local.add_channel('1H', v1=v1H)
    seqCP_local * rho

    # 13C acquisition
    rho.DetProp(seqAQ,n=n2)

    return np.asarray(rho.I[0])


# 2D HETCOR calculation

data = np.zeros((n1,n2), dtype=complex)

for i, t1 in enumerate(t1_axis):
    data[i, :] = simulate_t1_point(L=L, t1=t1, tCP=tCP, n2=n2)


# apodisation
window1 = np.exp(-3 * t1_axis / t1_axis[-1])
window2 = np.exp(-3 * t2_axis / t2_axis[-1])

data_windowed = data * window1[:, None] * window2[None, :]

spec = np.fft.fftshift(
    np.fft.ifft(data_windowed, axis=0),axes=0)
 
spec = np.fft.fftshift(np.fft.fft(spec, axis=1),axes=1)

spec_real = np.real(spec)

f1 = np.fft.fftshift(np.fft.fftfreq(n1, d=dt1))
f2 = np.fft.fftshift(np.fft.fftfreq(n2, d=dt2))

ppm1, ppm2 = f1/v0H, f2/v0C

print("F1 bandwith:", np.round(ppm1.min(),1), "ppm to", np.round(ppm1.max()), " ppm")
print("F2 bandwith:", np.round(ppm2.min(),1), "ppm to", np.round(ppm2.max()), " ppm")

# plot

pt.figure(figsize=(8,6))

spec_real /= spec_real.max()
levels = np.geomspace(0.3, 1, 12)

pt.contour(ppm2, ppm1, spec_real, colors='steelblue', levels=levels)
pt.contour(ppm2, ppm1, spec_real, colors='tomato', levels=-levels[::-1])
pt.xlabel(r"$^{13}$C (ppm)")
pt.ylabel(r"$^{1}$H (ppm)")
pt.gca().invert_xaxis()
pt.gca().invert_yaxis()
pt.grid()

pt.tight_layout()
pt.show()