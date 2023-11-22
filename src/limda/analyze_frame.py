import numpy as np
from collections import deque

from .neighbor import get_neighbor_list_using_cython
from .analyze_mols import get_mols_list_using_cython


class AnalyzeFrame:
    def __init__(self):
        pass

    # ---------------------------------------------------------------------------------------------------------
    def get_neighbor_list(
        self, mode: str, cut_off: float = None, bond_length: list[list[float]] = None
    ) -> list[list[int]]:
        """neighbor list を作成する
        Parameters
        ----------
            mode: str
                "bond_length"または"cut_off"
                mode = "bond_length"とした場合はneighbor listを結合種の長さ(bond_length)によって作成する
                mode = "cut_off"とした場合はneighbor listをカットオフによって作成する
            cut_off: float
                カットオフ半径
            bond_length: list[list[float]]
                結合の長さ
        """
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
    ) -> list[list[int]]:
        """分子ごとに原子のidを取得する
        例えば、水分子が3個とアンモニアが1個あるときは
        [[0, 1, 2],  # 水分子
        [3, 4, 5],  # 水分子
        [6, 7, 8],  # 水分子
        [9, 10, 11, 12]] # アンモニア
        Parameters
        ----------
            mode: str
                "bond_length"または"cut_off"
                mode = "bond_length"とした場合はneighbor listを結合種の長さ(bond_length)によって作成する
                mode = "cut_off"とした場合はneighbor listをカットオフによって作成する
            cut_off: float
                カットオフ半径
            bond_length: list[list[float]]
                結合の長さ
        """
        neighbor_list = self.get_neighbor_list(
            mode=mode, cut_off=cut_off, bond_length=bond_length
        )
        return get_mols_list_using_cython(neighbor_list, self.get_total_atoms())

    def get_mols_dict(
        self,
        mode: str = "bond_length",
        cut_off: float = None,
        bond_length: list[list[float]] = None,
    ) -> dict[str, list[list[int]]]:
        """分子ごとに原子のidを取得する
        例えば、水分子が3個とアンモニアが1個あるときは
        {"H2O1":[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
         "H3N1":[[9, 10, 11, 12]]}
        Parameters
        ----------
            mode: str
                "bond_length"または"cut_off"
                mode = "bond_length"とした場合はneighbor listを結合種の長さ(bond_length)によって作成する
                mode = "cut_off"とした場合はneighbor listをカットオフによって作成する
            cut_off: float
                カットオフ半径
            bond_length: list[list[float]]
                結合の長さ
        """

        mols_list = self.get_mols_list(
            mode=mode, cut_off=cut_off, bond_length=bond_length
        )
        mols_dict_tmp: dict[tuple(int), list[list[int]]] = {}
        atom_types: np.ndarray[int] = self.atoms["type"].values

        for mol in mols_list:
            atom_type_count: list[int] = [
                0 for _ in range(len(self.atom_type_to_symbol))
            ]
            for atom_idx in mol:
                atom_type_count[atom_types[atom_idx] - 1] += 1
            atom_type_count_tuple = tuple(atom_type_count)
            if atom_type_count_tuple not in mols_dict_tmp:
                mols_dict_tmp[atom_type_count_tuple] = []
            mols_dict_tmp[atom_type_count_tuple].append(mol)

        mols_dict: dict[str, list[list[int]]] = {}
        for atom_type_count, mols in mols_dict_tmp.items():
            mol_str = ""
            for atom_type in range(len(self.atom_type_to_symbol)):
                if atom_type_count[atom_type] == 0:
                    continue
                mol_str += f"{self.atom_type_to_symbol[atom_type + 1]}{atom_type_count[atom_type]}"

            mols_dict[mol_str] = mols

        return mols_dict

    def count_mols(
        self,
        mode: str = "bond_length",
        cut_off: float = None,
        bond_length: list[list[float]] = None,
    ) -> dict[str, int]:
        """分子数を数える
        例えば、水分子が3個とアンモニアが1個あるときは
        {"H2O1":3,
         "H3N1":1}
        Parameters
        ----------
            mode: str
                "bond_length"または"cut_off"
                mode = "bond_length"とした場合はneighbor listを結合種の長さ(bond_length)によって作成する
                mode = "cut_off"とした場合はneighbor listをカットオフによって作成する
            cut_off: float
                カットオフ半径
            bond_length: list[list[float]]
                結合の長さ
        """
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
                mol_str += f"{self.atom_type_to_symbol[atom_type + 1]}{atom_type_count[atom_type]}"

            mols_count[mol_str] = count

        return mols_count

    def count_bonds(
        self,
        mode: str = "bond_length",
        cut_off: float = None,
        bond_length: list[list[float]] = None,
    ) -> dict[str, int]:
        """結合数を数える
        例えば、水分子が3個あるときは
        {"H-O": 9, "H-H": 0, "O-O": 0}
        Parameters
        ----------
            mode: str
                "bond_length"または"cut_off"
                mode = "bond_length"とした場合はneighbor listを結合種の長さ(bond_length)によって作成する
                mode = "cut_off"とした場合はneighbor listをカットオフによって作成する
            cut_off: float
                カットオフ半径
            bond_length: list[list[float]]
                結合の長さ
        """
        neighbor_list = self.get_neighbor_list(
            mode=mode, cut_off=cut_off, bond_length=bond_length
        )
        atom_types = self.atoms["type"].values
        count_bonds_list = [
            [0 for _ in range(len(self.atom_symbol_to_type))]
                for _ in range(len(self.atom_symbol_to_type))
        ]
        for atom_i_idx in range(self.get_total_atoms()):
            atom_i_type = atom_types[atom_i_idx]
            for atom_j_idx in neighbor_list[atom_i_idx]:
                if atom_i_idx < atom_j_idx:
                    atom_j_type = atom_types[atom_j_idx]
                    count_bonds_list[atom_i_type - 1][atom_j_type - 1] += 1
        count_bonds_dict = {}
        for atom_i_type in range(1, len(self.atom_symbol_to_type) + 1):
            for atom_j_type in range(atom_i_type, len(self.atom_symbol_to_type) + 1):
                bond = f"{self.atom_type_to_symbol[atom_i_type]}-{self.atom_type_to_symbol[atom_j_type]}"
                count_bonds_dict[bond] = count_bonds_list[atom_i_type - 1][
                    atom_j_type - 1
                ]
        return count_bonds_dict

    def get_edge_index(self, cut_off: float) -> list[list[int]]:
        """allegroのedge_indexを作成します。
        edge_index : list[list[int]]でshapeは[2, num_edges]
                     原子i -> 原子j のみ(i < j)はいっていて、原子j -> 原子i は入っていない
        Parameters
        ----------
        cut_off: float
            edgeとしてみなす最大距離
        """
        neighbor_list = self.get_neighbor_list(mode="cut_off", cut_off=cut_off)
        edge_index = [[], []]
        for atom_idx in range(self.get_total_atoms()):
            for neighbor_atom_idx in neighbor_list[atom_idx]:
                if atom_idx < neighbor_atom_idx:
                    edge_index[0].append(atom_idx)
                    edge_index[1].append(neighbor_atom_idx)
        return edge_index
#----------------------------------------------------------------------------------------
    def get_sum_of_momentums(self)->np.ndarray[float]:
        """
        各方向の運動量の合計を計算する.

        Return
        ------
            momentum_sum : np.ndarray[float]
                運動量の合計 [x, y, z]
        """
        mass = np.array([self.atom_type_to_mass[typ] for typ in self.atoms["type"]])
        momentums = np.array([self.atoms["vx"], self.atoms["vy"], self.atoms["vz"]]) * mass
        return np.sum(momentums, axis=1)
        
