import sys
import pytest
import os
import adapt
from adapt.driver import *
from adapt.pyscf_backend import *
from adapt.of_translator import *


separation=float(sys.argv[1])
"""Test ADAPT on H4."""
if os.path.exists('test') == False:
    os.makedirs('test')
# geom = 'H 0 0 0; H 0 0 1; H 0 0 2; H 0 0 3'

geom=f'Be 0 0 0; H 0 0 -{separation}; H 0 0 {separation}'

# geom=f'Li 0 0 0; H 0 0 {separation}'

basis = "sto-3g"
reference = "rhf"

#Compute molecular integrals, etc.
E_nuc, H_core, g, D, C, hf_energy = get_integrals(geom, basis, reference, chkfile = "scr.chk", read = False)

#Compute number of electrons
N_e = int(np.trace(D))

#Compute sparse matrix reps
H, ref, N_qubits, S2, Sz, Nop = of_from_arrays(E_nuc, H_core, g, N_e)

#Build system data and get pool from it
s = sm.system_data(H, ref, N_e, N_qubits)
pool, v_pool = s.uccsd_pool(approach = 'vanilla')

ansatz=list(range(len(pool)))
params=np.zeros(len(ansatz))

for i, label in enumerate(v_pool):
    print(i, label)

state=t_ucc_state(params, ansatz, pool, ref)
E_init=t_ucc_E(params, ansatz, H, pool, ref)
print("Initial t-UCC energy:", E_init)

#Build 'xiphos' object (essentially ADAPT class)
xiphos = Xiphos(H, ref, "test", pool, v_pool, sym_ops = {"H": H, "S_z": Sz, "S^2": S2, "N": Nop})
params = np.array([])

error =xiphos.gd_adapt(params, ansatz, ref)

print("Final energy:", error)