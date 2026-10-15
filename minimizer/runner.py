# ref: https://github.com/giacomo-janson/sam2 (Apache-2.0, see LICENSE): sam/minimizer/runner.py, with aSAM's
# AllAtomProteinDataset replaced by topology.load_atom14 (same atom14 input, same batches of frames in file order)
import os
import yaml
import mdtraj
import torch

from . import get_topology, initialize, minimize, reconstruct_atom14
from .topology import get_traj_list, load_atom14


class Minimizer:

    def __init__(self,
            name: str,
            top_fp: str,
            ens_fp: str,
            protocol: str,
            params_fp: str = None
        ):

        # Load some data.
        self.name = name
        self.positions, self.aatype = load_atom14(top_fp, ens_fp)

        # Load the minimization parameters.
        module_dp = os.path.dirname(__file__)
        if protocol == "atlas":
            params_fp = os.path.join(module_dp, "params", "mizu_cfg.atlas.yaml")
        elif protocol == "mdcath":
            params_fp = os.path.join(module_dp, "params", "mizu_cfg.mdcath.yaml")
        elif protocol == "custom":
            if params_fp is None:
                raise ValueError()
        else:
            raise KeyError(protocol)
        with open(params_fp, 'r') as i_fh:
            params = yaml.safe_load(i_fh)
        self.opt_params = params["opt"]
        if "opt_ini" in params:
            self.opt_ini_params = params["opt_ini"]
        else:
            self.opt_ini_params = None
        self.top_params = params["top"]
        self.data_params = params.get("data", {"batch_size": 50})


    def run(self,
            batch_size: int = None,
            device: str = "cpu",
            verbose: bool = True
        ):
        # Setup the batch size.
        if batch_size is None:
            batch_size = self.data_params["batch_size"]

        # Iterate over the whole ensemble in batches.
        minimized_traj = []
        for i, start in enumerate(range(0, len(self.positions), batch_size)):

            # Get the xyz coordinates and the amino acid sequence.
            positions = self.positions[start:start + batch_size]
            a = self.aatype[None].repeat(len(positions), 1)
            positions = positions*0.1
            positions = positions.to(device)
            a = a.to(device)

            # Initialize.
            if i == 0:
                topology = get_topology(a, **self.top_params)
            positions = initialize(positions, a)

            # Brief initial minimization, typically with simple GD or Adam.
            if self.opt_ini_params is not None:
                positions, es = minimize(
                    positions=positions,
                    topology=topology,
                    return_early_stopping=True,
                    verbose=verbose,
                    **self.opt_ini_params
                )
                if not es:
                    positions = torch.autograd.Variable(positions)
                    positions.requires_grad = True
            else:
                es = False

            # Main minimization, typically with L-BFGS.
            if not es:
                # Run only if did not early-stop at the initial minimization.
                positions = minimize(
                    positions=positions,
                    topology=topology,
                    verbose=verbose,
                    **self.opt_params
                )

            # Reconstruct atom14 trajectory.
            atom14_rec = reconstruct_atom14(positions, topology)
            traj = get_traj_list({"positions": atom14_rec, "a": a}, join=True)
            minimized_traj.append(traj)

        # Join the mdtraj trajectories and return.
        minimized_traj = mdtraj.join(minimized_traj)

        return minimized_traj
