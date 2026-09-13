import numpy as np
import SLEEPY as sl
from ase import Atoms


def haeberlen(ms):
    vals = np.linalg.eigvalsh(ms)
    iso = vals.mean()
    d = vals - iso

    idx = np.argsort(np.abs(d))
    yy = vals[idx[0]]
    xx = vals[idx[1]]
    zz = vals[idx[2]]

    delta = zz - iso
    eta = (yy - xx) / delta if abs(delta) > 1e-12 else 0.0

    return delta, eta


def read_magres(filename):
    lattice = None
    atoms = []
    ms = {}

    with open(filename) as f:
        for line in f:
            if line.startswith("lattice"):
                lattice = np.array(
                    list(map(float, line.split()[1:10]))
                ).reshape(3, 3)

            elif line.startswith("atom "):
                s = line.split()
                atoms.append({
                    "element": s[1],
                    "index": int(s[3]),
                    "xyz": np.array(list(map(float, s[4:7]))),
                })

            elif line.startswith("ms "):
                s = line.split()
                ms[(s[1], int(s[2]))] = np.array(
                    list(map(float, s[3:12]))
                ).reshape(3, 3)

    return lattice, atoms, ms


def build_pair_spin_systems(
    filename,
    elem1,
    elem2,
    nuc1,
    nuc2,
    cutoff=2.0,
    v0H=400,
):
    lattice, atoms, ms = read_magres(filename)

    atoms1 = [a for a in atoms if a["element"] == elem1]
    atoms2 = [a for a in atoms if a["element"] == elem2]

    systems = []

    for a1 in atoms1:
        for a2 in atoms2:
            cell = Atoms(
                positions=[a1["xyz"], a2["xyz"]],
                cell=lattice,
                pbc=True,
            )
            r_angstrom = cell.get_distance(0, 1, mic=True)
            r_nm = r_angstrom / 10.0

            if r_angstrom > cutoff:
                continue

            ex = sl.ExpSys(v0H=v0H, Nucs=[nuc1, nuc2])

            # CSA on spin 1
            d, eta = haeberlen(ms[(elem1, a1["index"])])
            ex.set_inter("CSA", i=0, delta=d, eta=eta)

            # CSA on spin 2
            d, eta = haeberlen(ms[(elem2, a2["index"])])
            ex.set_inter("CSA", i=1, delta=d, eta=eta)

            # Dipolar coupling
            ex.set_inter(
                "dipole",
                i0=0,
                i1=1,
                delta=sl.Tools.dipole_coupling(r_nm, nuc1, nuc2),
            )

            systems.append({
                "pair": (
                    (elem1, a1["index"]),
                    (elem2, a2["index"]),
                ),
                "distance": r_angstrom,
                "ExpSys": ex,
            })

    return systems
