import numpy as np
from collections import deque

from .neighbor import get_neighbor_list_using_cython
from .analyze_mols import get_mols_list_using_cython


class AnalyzeFrame:
    def __init__(self):
        pass

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
            assert len(
                bond_length) == atom_type_num, "Incorrect format of bond length"
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
                count_bonds_dict[bond] = count_bonds_list[atom_i_type - 1][atom_j_type - 1]
                if atom_i_type != atom_j_type:
                    count_bonds_dict[bond] += count_bonds_list[atom_j_type - 1][atom_i_type - 1]
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

    def get_sum_of_momentums(self) -> np.ndarray[float]:
        """
        各方向の運動量の合計を計算する.

        Return
        ------
            momentum_sum : np.ndarray[float]
                運動量の合計 [x, y, z]
        """
        mass = np.array([self.atom_type_to_mass[typ]
                        for typ in self.atoms["type"]])
        momentums = np.array(
            [self.atoms["vx"], self.atoms["vy"], self.atoms["vz"]]) * mass
        return np.sum(momentums, axis=1)

    def get_neighbor_list_brute(self, bond_length: list[list[float]]) -> list[list[int]]:
        """ neighbor_listを作成します。
            pythonでO(N^2)のため、get_neighbor_list()のtest用です。
        Parameters
        -----------
            bond_length: list[list[float]]
        """
        neighbor_list_brute = [[] for _ in range(len(self))]
        x = self.atoms['x'].values
        y = self.atoms['y'].values
        z = self.atoms['z'].values
        atom_types = self.atoms['type'].to_list()
        for i in range(self.get_total_atoms()):
            for j in range(i+1, self.get_total_atoms()):
                dx: list[float] = [None, None, None]
                dx[0] = x[j] - x[i]
                dx[1] = y[j] - y[i]
                dx[2] = z[j] - z[i]
                for ax in range(3):
                    if dx[ax] < -self.cell[ax]/2:
                        dx[ax] += self.cell[ax]
                    elif self.cell[ax]/2 < dx[ax]:
                        dx[ax] -= self.cell[ax]
                if dx[0]*dx[0] + dx[1]*dx[1] + dx[2]*dx[2] <= bond_length[atom_types[i]-1][atom_types[j]-1]*bond_length[atom_types[i]-1][atom_types[j]-1]:
                    neighbor_list_brute[i].append(j)
                    neighbor_list_brute[j].append(i)

        for idx in range(len(neighbor_list_brute)):
            neighbor_list_brute[idx] = sorted(neighbor_list_brute[idx])

        return neighbor_list_brute

    def get_bond_length(self, cut_off: float) -> list[list[float]]:

        """
        neighbor_listを利用してcutoff内の2原子間の距離を結合種ごとに分類します。

        Return
        ------
            Dataset_radial_distribution_checker(config)に各原子組の距離が格納されるようにする。
            C-C: 1.8000, 1.9800, 1.6000...
            C-H: 1.1000, 1.2500, 1.0500...
            .
            .
            .
            Fe-Fe: 2.5000, 2.80000, 2.4600....
        """

        neighbor_list = self.get_neighbor_list(
            mode="cut_off", cut_off=cut_off,
        )
        atom_types = self.atoms["type"].values
        bond_length_list = [
            [[] for _ in range(len(self.atom_symbol_to_type))] for _ in range(len(self.atom_symbol_to_type))
        ]
        for atom_i_idx in range(self.get_total_atoms()):
            atom_i_type = atom_types[atom_i_idx]
            for atom_j_idx in neighbor_list[atom_i_idx]:
                if atom_i_idx < atom_j_idx:
                    atom_j_type = atom_types[atom_j_idx]
                    ij_vector: list[float] = [self.atoms["x"][atom_i_idx] - self.atoms["x"][atom_j_idx],
                                              self.atoms["y"][atom_i_idx] - self.atoms["y"][atom_j_idx],
                                              self.atoms["z"][atom_i_idx] - self.atoms["z"][atom_j_idx]]
                    for ax in range(3):
                        if ij_vector[ax] < -self.cell[ax]/2:
                            ij_vector[ax] += self.cell[ax]
                        elif ij_vector[ax] > self.cell[ax]/2:
                            ij_vector[ax] -= self.cell[ax]
                        
                    bond_length_i_and_j = np.sqrt(ij_vector[0]**2 + ij_vector[1]**2 + ij_vector[2]**2)
                    #print(bond_length_i_and_j)
                    assert bond_length_i_and_j < cut_off + 0.1 , "WARNING: The algorithm is wrong."
                    bond_length_list[atom_i_type - 1][atom_j_type - 1].append(bond_length_i_and_j)
                        
        
        bond_length_dict: dict[str, list[float]] = {}
        for atom_i_type in range(1,len(self.atom_symbol_to_type)+1):
            for atom_j_type in range(1,len(self.atom_symbol_to_type)+1):
                if atom_i_type < atom_j_type:
                    bond = f"{self.atom_type_to_symbol[atom_i_type]}-{self.atom_type_to_symbol[atom_j_type]}"
                    if bond in bond_length_dict:
                        bond_length_dict[bond] += bond_length_list[atom_i_type - 1][atom_j_type - 1]
                    else:
                        bond_length_dict[bond] = bond_length_list[atom_i_type - 1][atom_j_type - 1]
                elif atom_i_type >= atom_j_type:
                    bond = f"{self.atom_type_to_symbol[atom_j_type]}-{self.atom_type_to_symbol[atom_i_type]}"
                    if bond in bond_length_dict:
                        bond_length_dict[bond] += bond_length_list[atom_i_type - 1][atom_j_type - 1]
                    else:
                        bond_length_dict[bond] = bond_length_list[atom_i_type - 1][atom_j_type - 1]

        """sum_list = 0
        for i in range(len(bond_length_list)):
            for j in range(len(bond_length_list[i])):
                a = len(bond_length_list[i][j])
                print(f"{i}-{j} {a}")
                sum_list += a
        print(f"sum ={sum_list}")

        sum_dict = 0
        for key in bond_length_dict.keys():
            a = len(bond_length_dict[key])
            print(f"{key} {a}")
            sum_dict += a
        print(f"dict sum = {sum_dict}")"""
        return bond_length_dict
    
    def get_bond_length_group(self, cut_off: float, interval: float) -> list[list[float]]:
        """
        neighbor_listを利用してcutoff内の2原子間の距離を結合種ごとに分類します。

        Return
        ------
            get_bond_lengthから得られた配列
            C-C: 1.8000, 1.9800, 1.6000...
            C-H: 1.1000, 1.2500, 1.0500...
            .
            .
            .
            Fe-Fe: 2.5000, 2.80000, 2.4600....
            をintervalごとに要素数を数える
            C-C: 0, 0, 0, 10, 1400, 20000, ... 0, 0 
            C-H: 0, 0, 3, 40, 5000, 64000, ... 0, 0
            .
            .
            .
            Fe-Fe: 0, 0, 0, 0, 20, 300, ... 400, 0
        """
        bond_length_dict = self.get_bond_length(cut_off=cut_off)
        #print(bond_length_dict)
        #exit()
        bond_length_group_dict: dict[str, list[int]] = {}

        group_num = int(np.ceil(cut_off/interval))
        for bond_key in bond_length_dict.keys():
            bond_length_group = np.zeros(group_num, dtype=int)
            for bond_length in bond_length_dict[bond_key]:
                for group_no in range(group_num):
                    #print(group_no*interval)
                    if group_no* interval <= bond_length and (group_no+1)*interval > bond_length:
                        bond_length_group[group_no] += 1
            bond_length_group_dict[bond_key] = bond_length_group
        
        return bond_length_group_dict