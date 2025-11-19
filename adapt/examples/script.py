import argparse
import numpy as np
import pandas as pd
import logging
from adapt.pyscf_backend import get_integrals, get_F
from adapt.of_translator import of_from_arrays
import adapt.system_methods as sm
from adapt.driver import t_ucc_E, Xiphos


#     E_nuc, H_core, g, D, C, hf_energy = get_integrals(geom, basis, reference, chkfile="scr.chk", read=False, active=active)
#     N_e = int(np.trace(D))
#     H, ref, N_qubits, S2, Sz, Nop = of_from_arrays(E_nuc, H_core, g, N_e)

argparser = argparse.ArgumentParser(description='Run Xiphos VQE example.')

argparser.add_argument('--mol', type=str, default='h4', help='Molecule to simulate (h, lih, beh, n).')
argparser.add_argument('--r', type=float, default=1.326, help='Bond length for the molecule.')

argparser.add_argument('--steps', type=int, default=10, help='Number of adiabatic steps.')

args = argparser.parse_args()

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
        elif self.mol=='h4':
            geom = f'H 0 0 0; H 0 0 {self.bond_length}; H 0 {self.bond_length} 0; H 0 1 {self.bond_length}'
            basis = "sto-3g"
        else:
            raise ValueError("Molecule not recognized. Choose from: h, lih, beh, n, h4")
        self.geom = geom
        self.basis = basis
        self.active = active
        self.r=bond_length
        self.E_nuc, self.H_core, self.g, self.D, self.C, self.hf_energy, self.fci_energy= get_integrals(self.geom, self.basis, "rhf", chkfile=f"scr{self.r}.chk", read=False, active=self.active)
        self.N_e = int(np.trace(self.D))
        self.H, self.ref, self.N_qubits, self.S2, self.Sz, self.Nop = of_from_arrays(self.E_nuc, self.H_core, self.g, self.N_e) 
        self.F_dense=get_F(self.geom, self.basis,"rhf",sparse=True,qubits=self.N_qubits,)
        self.F, _ , _, _, _ ,_=of_from_arrays(self.hf_energy, self.F_dense, np.zeros_like(self.g), self.N_e)

        

class Xiphos_Builder:
    def __init__(self, molecular: Molecular):
        self.molecular = molecular
        self.F=self.molecular.F
        self.s = sm.system_data(self.molecular.H, self.molecular.ref, self.molecular.N_e, self.molecular.N_qubits)
        self.jw_pool, self.v_pool = self.s.uccsd_pool(approach="vanilla")
        print(f"Number of qubits: {self.molecular.N_qubits}, Number of operators in pool: {len(self.jw_pool)}")
        self.ansatz = list(range(len(self.jw_pool)))
        self.params = np.zeros(len(self.ansatz))
        self.xi=Xiphos(self.molecular.H, self.molecular.ref, "test", self.jw_pool, self.v_pool, sym_ops={"H": self.molecular.H, "S_z": self.molecular.Sz, "S^2": self.molecular.S2, "N": self.molecular.Nop})

  
    def gd_detailed(self):
        xi=self.xi
        [res, string], vqe_energies=xi.gd_detailed_vqe(self.params, self.ansatz, seed=32)
        return [res, string], vqe_energies

    def aavqe(self,steps):
        xi=self.xi
        res, vqe_energies=xi.gd_adiabatic_vqe(self.params, self.ansatz, self.F, steps=steps)
        return res, vqe_energies

    

if __name__ == "__main__":
    r=float(args.r)
    mol=args.mol
    steps=int(args.steps)
    mol_instance=Molecular(mol, r)

    print(mol_instance.active)
    xiphos_builder=Xiphos_Builder(mol_instance)

    result, vqe_energies= xiphos_builder.aavqe(steps)
    print("vqe energies:", vqe_energies)
    # # string=[float(x) for x in string]
    # print(f"Final energy: {res} Ha")

    # print("Operator string:", string)
    # print("VQE energies at each step:", vqe_energies)
    
    # with open(f'string{r}_{mol_instance.mol}.txt', 'w') as f:
    #     for item in string:
    #         f.write(f"{item}")
    df = pd.DataFrame(vqe_energies)
    # print(df.tail())
    import pandas as pd

    # expand the list column into multiple columns
    expanded = df["vqe_energies"].apply(pd.Series)

    # rename columns
    expanded.columns = [f"energy_{i}" for i in expanded.columns]

    # join back to original df
    df_expanded = pd.concat([df[["t"]], expanded], axis=1)
    # print(df_expanded.head())
    records = (
        df.apply(lambda row: [
            {"t": row["t"], "epoch": i, "energy": e}
            for i, e in enumerate(row["vqe_energies"])
        ], axis=1)
        .explode()
        .tolist()
    )

    df_long = pd.DataFrame(records)
    df_long['etime'] = df_long['t'] + (df_long['epoch'] / df_long['epoch'].max() - 1) * (1 / steps)
    
    
    df_long=df_long[['t','epoch','etime','energy']]

    df=df_long.copy()
    df['HF Energy (Ha)'] = mol_instance.hf_energy
    df['FCI Energy (Ha)']=mol_instance.fci_energy
    df['r']=r
    df['energy_error']=df['energy'] - mol_instance.fci_energy
    print(df)
 
    df.to_csv(f'aavqe_energies_{r}_{steps}.csv', index=False)
