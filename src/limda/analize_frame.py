import numpy as np
from collections import deque

from .neighbor import get_neighbor_list_using_cython
from .analyze_mols import get_mols_list_using_cython


class AnalizeFrame:
    def __init__(self):
        pass

    # ---------------------------------------------------------------------------------------------------------
    def get_neighbor_list(
        self, mode: str, cut_off: float = None, bond_length: list[list[float]] = None
    ) -> list[list[int]]:
        assert mode == "bond_length" or mode == "cut_off", "Please configure mode"
        atom_type_num = len(self.atom_symbol_to_type)
        if mode == "bond_length":
            if bond_length is None:
                if "bond_length" in self.limda_default:
                    bond_length = self.limda_default["bond_length"]
            assert len(bond_length) == atom_type_num, "Incorrect format of bond length"
            for bond_list in bond_length:
                assert (
                    len(bond_list) == atom_type_num
                ), "Incorrect format of bond length"
        elif mode == "cut_off":
            if cut_off is None:
                if "cut_off" in self.limda_default:
                    cut_off = self.limda_default["cut_off"]
            bond_length = [
                [cut_off for _ in range(atom_type_num)] for __ in range(atom_type_num)
            ]

        mesh_length = (
            max(list(map(lambda x: max(x), bond_length))) + 0.01
        )  # cut_off(bond_length) + margin
        if mesh_length * 3 > min(self.cell):
            mesh_length = min(self.cell) / 3

        neighbor_list = get_neighbor_list_using_cython(
            atoms_type=self.atoms["type"],
            atoms_pos=[self.atoms["x"], self.atoms["y"], self.atoms["z"]],
            mesh_length=mesh_length,
            atom_num=len(self),
            bond_length=bond_length,
            cell=self.cell,
        )
        return neighbor_list

    def get_mols_list(
        self,
        mode: str = "bond_length",
        cut_off: float = None,
        bond_length: list[list[float]] = None,
    ):
        neighbor_list = self.get_neighbor_list(
            mode=mode, cut_off=cut_off, bond_length=bond_length
        )
        return get_mols_list_using_cython(neighbor_list, self.get_total_atoms())

    def count_mols(
        self,
        mode: str = "bond_length",
        cut_off: float = None,
        bond_length: list[list[float]] = None,
    ):
        mols_list = self.get_mols_list(
            mode=mode, cut_off=cut_off, bond_length=bond_length
        )
        mols_count_tmp: dict[tuple(int), int] = {}
        atom_types: np.ndarray[int] = self.atoms["type"].values

        for mol in mols_list:
            atom_type_count: list[int] = [
                0 for _ in range(len(self.atom_type_to_symbol))
            ]
            for atom_idx in mol:
                atom_type_count[atom_types[atom_idx] - 1] += 1
            atom_type_count_tuple = tuple(atom_type_count)
            if atom_type_count_tuple not in mols_count_tmp:
                mols_count_tmp[atom_type_count_tuple] = 0
            mols_count_tmp[atom_type_count_tuple] += 1

        mols_count: dict[str, int] = {}
        for atom_type_count, count in mols_count_tmp.items():
            mol_str = ""
            for atom_type in range(len(self.atom_type_to_symbol)):
                if atom_type_count[atom_type] == 0:
                    continue
                if mol_str == "":
                    mol_str = f"{self.atom_type_to_symbol[atom_type + 1]}{atom_type_count[atom_type]}"
                else:
                    mol_str += f" {self.atom_type_to_symbol[atom_type + 1]}{atom_type_count[atom_type]}"

            mols_count[mol_str] = count

        return mols_count
