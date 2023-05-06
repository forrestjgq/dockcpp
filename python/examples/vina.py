import pydock

cmd="--receptor 1iep_receptor.pdbqt --ligand 1iep_ligand.pdbqt --out ligs_out.pdbqt --center_x 15.190 --center_y 53.903 --center_z 16.917 --size_x 20 --size_y 20 --size_z 20 --scoring vina --num_modes 10 --seed 123"
pydock.vina(cmd)