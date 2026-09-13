import SLEEPY as sl
import numpy as np
import matplotlib.pyplot as pt

# Experimental conditions

v0H = 600          # 1H frequency in MHz
v0C = v0H * 0.25144953
vr = 60e3          # MAS frequency in Hz
# NOTE: at 20 kHz MAS, the ~47 kHz CH dipolar coupling below is not well
# averaged. Over the longer t1 windows needed to resolve small 1H shifts
# (see SW1 below), that un-averaged coupling dominates the CP transfer and
# drags the F2 peak away from the true 13C shift. Fast MAS (>~ the coupling
# strength) is required to keep t1 evolution "clean" (close to pure 1H
# chemical-shift precession).

# Powder averaging
pwdavg = 2
n_gamma = 10

# CP
v1H = 60e3         # 1H RF field, Hz
v1C = 60e3         # 13C RF field, Hz
tCP = 0.5e-3       # CP contact time, 0.5 ms

# Direct (13C) acquisition
SW1 = 10e3         # indirect (1H) dimension bandwidth, Hz -- sized to the
                   # actual 1H shift range so shifts don't alias near F1=0
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

def simulate_t1_point(L, t1, tCP, n2, start='1Hx'):
    # Start with proton transverse magnetization.
    # 'start' selects the prep-pulse phase: '1Hx' -> cosine-modulated data,
    # '1Hy' -> sine-modulated data. Both are needed for F1 quadrature
    # (States) detection -- see simulation loop below.
    rho = sl.Rho(rho0=start, detect='13Cp')

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
# Two t1 series (cosine and sine) are simulated and combined into a
# complex (States) t1 interferogram so that the F1 (1H) dimension gets
# proper quadrature (sign-discriminated) detection. Without this, 1Hx
# alone amplitude-modulates the signal in t1 and produces a pair of
# mirror-image peaks at +F1 and -F1 after the Fourier transform.

data_cos = np.zeros((n1,n2), dtype=complex)
data_sin = np.zeros((n1,n2), dtype=complex)

for i, t1 in enumerate(t1_axis):
    data_cos[i, :] = simulate_t1_point(L=L, t1=t1, tCP=tCP, n2=n2, start='1Hx')
    data_sin[i, :] = simulate_t1_point(L=L, t1=t1, tCP=tCP, n2=n2, start='1Hy')

# States hypercomplex combination.
# NOTE: depending on SLEEPY's internal rotation-direction convention, your
# peak may land on the wrong side of F1=0. If so, flip the sign below to
# "data_cos - 1j*data_sin".
data = data_cos - 1j*data_sin


# apodisation
window1 = np.exp(-3 * t1_axis / t1_axis[-1])
window2 = np.exp(-3 * t2_axis / t2_axis[-1])

data_windowed = data * window1[:, None] * window2[None, :]

# zero-filling
# Pads the (apodised) time-domain data with zeros before the FFT. This adds
# no new information, but interpolates the spectrum onto a finer frequency
# grid -- useful for reading off peak positions accurately and for smoother-
# looking contours. Zero-fill AFTER windowing so the decaying window doesn't
# get corrupted by the appended zeros; a factor of 2-4x per dimension is
# typical.
zf1 = 4 * n1
zf2 = 2 * n2

spec = np.fft.fftshift(np.fft.fft2(data_windowed, s=(zf1, zf2), axes=(0,1)), axes=(0,1))
spec_abs = np.abs(spec)

f1 = np.fft.fftshift(np.fft.fftfreq(zf1, d=dt1))
f2 = np.fft.fftshift(np.fft.fftfreq(zf2, d=dt2))

ppm1, ppm2 = f1/v0H, f2/v0C

print("F1 bandwith:", np.round(f1.min()/v0H,1), "ppm to", np.round(f1.max()/v0H,1), " ppm")
print("F2 bandwith:", np.round(f2.min()/v0C,1), "ppm to", np.round(f2.max()/v0C,1), " ppm")

# plot

pt.figure(figsize=(8,6))

spec_abs /= spec_abs.max()
levels = np.geomspace(0.3, 1, 12)

pt.contour(ppm2, ppm1, spec_abs, levels=levels)
pt.xlabel(r"$^{13}$C (ppm)")
pt.ylabel(r"$^{1}$H (ppm)")
pt.gca().invert_xaxis()
pt.gca().invert_yaxis()

pt.tight_layout()
pt.savefig('hetcor_states.png', dpi=150)
pt.show()
