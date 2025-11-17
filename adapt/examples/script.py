import argparse
import numpy as np
import pandas as pd
# adapt package internals
from adapt.pyscf_backend import get_integrals
from adapt.of_translator import of_from_arrays
import adapt.system_methods as sm
from adapt.driver import t_ucc_E
def build_system(geom: str, basis: str = "sto-3g", reference: str = "rhf",active: tuple=None):
    E_nuc, H_core, g, D, C, hf_energy = get_integrals(geom, basis, reference, chkfile="scr.chk", read=False, active=active)
    N_e = int(np.trace(D))
    H, ref, N_qubits, S2, Sz, Nop = of_from_arrays(E_nuc, H_core, g, N_e)
    return H, ref, N_qubits, N_e

def main():
    p = argparse.ArgumentParser(description="Simple full-UCCSD VQE using adapt internals (default: BeH / sto-3g, optimize with L-BFGS-B)")
    p.add_argument("--molecule", type=str, help="Molecule name (h,lih,beh,n)")
    p.add_argument("--ref", type=str, default="rhf")
    p.add_argument("--no-optimize", action="store_true", help="Do not run the classical optimizer (run energy eval only)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--r", type=float, help="bond length" )
    # ignore extra Jupyter args when running inside notebooks
    args, _ = p.parse_known_args()
    if args.molecule=='h':
        geom = f'H 0 0 0; H 0 0 {args.r}'
        basis = "sto-3g"
    elif args.molecule=='lih':
        geom = f'Li 0 0 0; H 0 0 {args.r}'
        basis = "sto-3g"
    elif args.molecule=='beh':
        geom = f'Be 0 0 0; H 0 0 -{args.r}; H 0 0 {args.r}'
        basis = "sto-3g"
        active=(4,4)
    elif args.molecule=='n':
        geom = f'N 0 0 0; N 0 0 {args.r}'
        basis = "cc-pvdz"
    else:
        raise ValueError("Molecule not recognized. Choose from: h, lih, beh, n")
    H, ref, N_qubits, N_e = build_system(geom, basis=basis, reference=args.ref)

    s = sm.system_data(H, ref, N_e, N_qubits)
    jw_pool, v_pool = s.uccsd_pool(approach="vanilla")

    # full-UCCSD ansatz: use every operator in the pool in sequence
    ansatz = list(range(len(jw_pool)))
    params = np.zeros(len(ansatz))

    # evaluate initial energy
    E0 = t_ucc_E(params, ansatz, H, jw_pool, ref)
    print(f"Initial (params=0) energy: {float(E0):.12f} Ha  |  #operators = {len(ansatz)}")

    if args.no_optimize:
        return

    try:
        from scipy.optimize import minimize
    except Exception:
        print("scipy not available: cannot optimize. Install scipy and retry.")
        return

    rng = np.random.default_rng(args.seed)
    # small random start (better than all-zeros to break possible stationary points)
    x0 = rng.normal(scale=0.05, size=len(ansatz))

    def obj(x):
        return float(t_ucc_E(x, ansatz, H, jw_pool, ref))
    energies=[]
    def mycallback(xk):
        e = obj(xk)
        energies.append(e)
    res = minimize(obj, x0, method="L-BFGS-B", options={"maxiter": 200},callback=mycallback)
    print("Optimizer success:", bool(res.success), res.message)
    print(f"Optimized energy: {res.fun:.12f} Ha")
    if len(res.x) > 0:
        print("First 8 parameters:", np.array2string(res.x[:8], precision=6, separator=", "))

    pd.DataFrame(energies, columns=["Energy"]).to_csv(f"{args.molecule}_uccsd_vqe_energies.csv", index=False)
if __name__ == "__main__":
    main()
