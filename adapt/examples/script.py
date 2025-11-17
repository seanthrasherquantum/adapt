import argparse
import numpy as np
import pandas as pd
# adapt package internals
from adapt.pyscf_backend import get_integrals
from adapt.of_translator import of_from_arrays
import adapt.system_methods as sm
from adapt.driver import t_ucc_E, Xiphos

#     E_nuc, H_core, g, D, C, hf_energy = get_integrals(geom, basis, reference, chkfile="scr.chk", read=False, active=active)
#     N_e = int(np.trace(D))
#     H, ref, N_qubits, S2, Sz, Nop = of_from_arrays(E_nuc, H_core, g, N_e)
#     return H, ref, N_qubits, N_e
class Molecular:
    def __init__(self, mol: str, bond_length: float,active : tuple=None):
        self.mol = mol
        self.bond_length = bond_length
        if self.mol=='h':
            geom = f'H 0 0 0; H 0 0 {self.bond_length}'
            basis = "sto-3g"
        elif self.mol=='lih':
            geom = f'Li 0 0 0; H 0 0 {self.bond_length}'
            basis = "sto-3g"
        elif self.mol=='beh':
            geom = f'Be 0 0 0; H 0 0 -{self.bond_length}; H 0 0 {self.bond_length}'
            basis = "sto-3g"
            active=(4,4)
        elif self.mol=='n':
            geom = f'N 0 0 0; N 0 0 {self.bond_length}'
            basis = "cc-pvdz"
        else:
            raise ValueError("Molecule not recognized. Choose from: h, lih, beh, n")
        self.geom = geom
        self.basis = basis
        self.active = active
        self.E_nuc, self.H_core, self.g, self.D, self.C, self.hf_energy = get_integrals(self.geom, self.basis, "rhf", chkfile="scr.chk", read=False, active=self.active)
        self.N_e = int(np.trace(self.D))
        self.H, self.ref, self.N_qubits, self.S2, self.Sz, self.Nop = of_from_arrays(self.E_nuc, self.H_core, self.g, self.N_e) 

class Xiphos_Builder:
    def __init__(self, molecular: Molecular):
        self.molecular = molecular
        self.s = sm.system_data(self.molecular.H, self.molecular.ref, self.molecular.N_e, self.molecular.N_qubits)
        self.jw_pool, self.v_pool = self.s.uccsd_pool(approach="vanilla")
        self.ansatz = list(range(len(self.jw_pool)))
        self.params = np.zeros(len(self.ansatz))
        self.xi=Xiphos(self.molecular.H, self.molecular.ref, "test", self.jw_pool, self.v_pool, sym_ops={"H": self.molecular.H, "S_z": self.molecular.Sz, "S^2": self.molecular.S2, "N": self.molecular.Nop})

  
    def gd_detailed(self):
        xi=self.xi
        [res, string], vqe_energies=xi.gd_detailed_vqe(self.params, self.ansatz, seed=32)
        return [res, string], vqe_energies

def run_example(separation):

    mol_instance=Molecular('h', separation)
    

    xiphos_builder=Xiphos_Builder(mol_instance)
    
    [res, string], vqe_energies= xiphos_builder.gd_detailed()
    # string=[float(x) for x in string]
    print(f"Final energy: {res} Ha")
    
    print("Operator string:", string)
    
    print("VQE energies at each step:", vqe_energies)

    with open(f'string{separation}_{mol_instance.mol}.txt', 'w') as f:
        for item in string:
            f.write(f"{item}")
    
    pd.DataFrame(vqe_energies, columns=['VQE Energy (Ha)']).to_csv(f'vqe_energies_{separation}_{mol_instance.mol}.csv', index=False)
if __name__ == "__main__":

    run_example(0.74)