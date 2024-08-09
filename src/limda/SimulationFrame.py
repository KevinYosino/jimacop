import pandas as pd
import numpy as np
import random
from copy import deepcopy
import sys
import os
from typing import Any
from .import_frame import ImportFrame
from .export_frame import ExportFrame
from .calculate import Calculate
from .analyze_frame import AnalyzeFrame
from . import const as C


class SimulationFrame(
    ImportFrame,
    ExportFrame,
    Calculate,
    AnalyzeFrame,
):
    """シミュレーションしたデータを読み込み、書き込み、分析するためのクラス
    一つのフレームを扱う

    Attributes
    ----------
    atoms : pd.DataFrame
        原子のtype, 座標, 速度, 加速度, 力, type, NNPによる力の推論値などを含むpandasのDataFrame
        原子のタイプは1-indexed
    cell : np.array
        cellの大きさが入ったarray, shape:[3]
        cell[0]:x方向, cell[1]:y方向, cell[2]:z方向
    atom_symbol_to_type : dict[str, int]
        原子のシンボルをkey, 原子のtypeをvalueとするdict
    atom_type_to_symbol : dict[int, str]
        原子のtypeをkey, 原子のシンボルをvalueとするdict
    atom_type_to_mass : dict[int, float]
        原子のtypeをkey, 原子の質量(g/mol)をvalueとするdict
    step_num: int
        MDした時のこのSimulationFrameが持つステップ数
    potential_energy: float
        このフレームが持つポテンシャルエネルギー, 単位はeV
    pred_potential_energy: float
        NNPにより推論したときのポテンシャルエネルギー,単位はeV
    virial_tensor : np.array[float] 
        このフレームの持つvirialテンソル, 単位はeV, shape:(3,3)
    pred_virial_tensor : np.array[float] 
        NNPにより推論したときのvirialテンソル, 単位はeV, shape:(3,3)
    """
    atoms: pd.DataFrame
    cell: np.ndarray[float]  # shape:[3]
    atom_symbol_to_type: dict[str, int]
    atom_type_to_symbol: dict[int, str]
    atom_type_to_mass: dict[int, float]
    step_num: int
    potential_energy: float
    virial_tensor: np.ndarray[float]
    pred_potential_energy: float
    pred_virial_tensor: np.ndarray[float]
    limda_default: dict[str, Any]

    def __init__(self, para: str = ""):
        self.atoms = None
        self.cell = None
        self.atom_symbol_to_type = None
        self.atom_type_to_symbol = None
        self.atom_type_to_mass = None
        self.step_num = None
        self.potential_energy = None
        self.virial_tensor = None
        self.pred_potential_energy = None
        self.pred_virial_tensor = None
        self.import_limda_default()
        self.import_para_from_str(para)

    def __getitem__(self, key) -> pd.DataFrame:
        """
        sdat.atoms[column]をsdat[column]と省略して書くことが出来る。
        """
        return self.atoms[key]

    def __setitem__(self, key, val) -> None:
        self.atoms[key] = val

    def __len__(self) -> int:
        """
        sdat.get_total_atoms()をlen(sdat)と省略して書くことが出来る。
        """
        return self.get_total_atoms()

    def get_total_atoms(self) -> int:
        """
        全原子数を返す関数。
        """
        assert self.atoms is not None, 'Import file first'
        return len(self.atoms)

    def get_atom_type_set(self) -> set:
        """
        系内の原子のtypeのsetを返す関数
        """
        return set(self.atoms['type'])

    def wrap_atoms(self) -> None:
        """
        セルの外にはみ出している原子をセルの中に入れる。
        """
        assert self.cell is not None, "set sf.cell"
        assert 0 not in set(self.cell), "cell size must not be 0"

        self.atoms[['x', 'y', 'z']] %= self.cell

    def replicate_atoms(self, replicate_directions: list[int] = [1, 1, 1]) -> None:
        """
        x, y, z 方向にセルを複製する関数

        Parameters
        ----------
        replicate_directions : list
            x, y, z方向に何倍するかを指定する。
            例えばx方向に2倍,y方向に3倍,z方向に4倍したい時は
            replicate_directions = [2, 3, 4] とする

        """
        for i, dim in enumerate(['x', 'y', 'z']):
            copy_atoms = self.atoms.copy()
            for idx in range(1, replicate_directions[i]):
                def replicate(d): return d[dim] + self.cell[i]*idx
                append_atoms = copy_atoms.copy()
                append_atoms[dim] = append_atoms.apply(replicate, axis=1)
                self.atoms = pd.concat([self.atoms, append_atoms])

        self.atoms.reset_index(drop=True, inplace=True)
        for dim in range(3):
            self.cell[dim] *= replicate_directions[dim]

    def concat_atoms(self, outer_sf) -> None:
        """
        sfとouter_sfを結合する関数

        Parameters
        ----------
        outer_sf : SimulationFrame
            取り入れたいsfを指定する。

        """
        self.atoms = pd.concat([self.atoms, outer_sf.atoms])
        self.atoms.reset_index(drop=True, inplace=True)
        for dim in range(3):
            self.cell[dim] = max(self.cell[dim], outer_sf.cell[dim])

    def delete_atoms(self, condition, reindex):
        """
        条件に当てはまる原子を削除する

        Parameters
        ----------
        condition : function
            削除したい原子の条件を指定する関数
        reindex : bool
            reindex == Trueの時は原子のid(idx)は新しく割り振られる
            reindex == Falseの時は原子のid(idx)は新しく割り振られず、削除前のものと変わらない
        """
        if callable(condition):
            target_atoms = condition(self)
            self.atoms = self.atoms[~target_atoms]
        else:
            self.atoms = self.atoms[~condition]
        if reindex:
            self.atoms.reset_index(drop=True, inplace=True)

    def density(self, x_min=None, x_max=None, y_min=None, y_max=None, z_min=None, z_max=None):
        """セル内の密度を計算する関数
        Parameters
        ----------
        x_max: float
            xの上限
        x_min: float
            xの下限
        y_max: float
            yの上限
        y_min: float
            yの下限
        z_max: float
            zの上限
        z_min: float
            zの下限
        Returns
        -------
        density : float
            セル内の密度(g/cm^3)
        """
        x_mx = x_max if x_max is not None else self.cell[0]
        x_mn = x_min if x_min is not None else 0
        assert 0 <= x_mn and x_mx <= self.cell[0], 'set correct x_max and x_min'
        assert x_mx >= x_mn, 'set correct x_max and x_min'

        y_mx = y_max if y_max is not None else self.cell[1]
        y_mn = y_min if y_min is not None else 0
        assert 0 <= y_mn and y_mx <= self.cell[1], 'set correct y_max and y_min'
        assert y_mx >= y_mn, 'set correct y_max and y_min'

        z_mx = z_max if z_max is not None else self.cell[2]
        z_mn = z_min if z_min is not None else 0
        assert 0 <= z_mn and z_mx <= self.cell[2], 'set correct z_max and z_min'
        assert z_mx >= z_mn, 'set correct z_max and z_min'

        # 体積(cm^3)
        volume = (x_mx - x_mn) * (y_mx - y_mn) * (z_mx - z_mn) * (10 ** - 24)
        all_weight = 0

        def condition(sf):
            target_atoms = (x_mn <= sf.atoms['x']) & (sf.atoms['x'] <= x_mx)
            target_atoms &= (y_mn <= sf.atoms['y']) & (sf.atoms['y'] <= y_mx)
            target_atoms &= (z_mn <= sf.atoms['z']) & (sf.atoms['z'] <= z_mx)
            return target_atoms
        atom_type_counter = self.count_atom_types(
            res_type='dict', condition=condition)
        for atom_symbol, atom_type_count in atom_type_counter.items():
            atom_type = self.atom_symbol_to_type[atom_symbol]
            atom_mass = self.atom_type_to_mass[atom_type]
            all_weight += atom_type_count * atom_mass / C.AVOGADORO_CONST
        # セル内の密度(g/cm^3)
        density = all_weight / volume
        return density

    def count_atom_types(self, res_type='series', condition=None):
        """原子のタイプごとに原子の個数をカウントする関数
        Parameters
        ----------
        res_type : str
            res_type='series'のときは結果をpd.Seriesで返す
            res_type='dict'のときは結果をdictで返す
        condition : function
            condition関数
        """
        if condition is None:
            target_atoms = np.array([True] * self.get_total_atoms())
        else:
            target_atoms = condition(self)
        if res_type == 'series':
            return self.atoms.loc[target_atoms, 'type'].value_counts().rename(index=self.atom_type_to_symbol)
        elif res_type == 'dict':
            return self.atoms.loc[target_atoms, 'type'].value_counts().rename(index=self.atom_type_to_symbol).to_dict()
        else:
            raise ValueError(
                f'res_type: {res_type} is not supported. supported res_type : [series, dict]')

    def shuffle_type(self, type_ratio: list[float], fix_type: list[int] = None):
        """sfのtypeをランダムにシャッフルする。
            atomsに座標を持たせてから使用。
        Parameters
        ----------
            type_ratio: list[float]
                typeに対する割合が入ったlist

            fix_type: list[int]
                固定するタイプが入ったlist
        Example
        -------
            sf.shuffle_type([1,2,3,0])
                原子数:6 -> sf.atoms["type"] = [1,2,2,3,3,3] をシャッフルしたもの
                余りはtype_ratioに応じてランダムに入る
        """
        if fix_type is not None:
            atoms_tmp = self.atoms.query('type in @fix_type')
            self.atoms = self.atoms.query('type not in @fix_type')

        tot_atoms = self.get_total_atoms()
        tot_ratio = sum(type_ratio)
        type_ratio = [(type_ratio[i]*tot_atoms/tot_ratio)
                      for i in range(len(type_ratio))]
        remain_weight = [type_ratio[i] -
                         int(type_ratio[i]) for i in range(len(type_ratio))]
        type_list = []
        for idx, ratio in enumerate(type_ratio):
            type_list.extend([idx+1 for _ in range(int(ratio))])

        if len(type_list) != len(self):
            remain_type = random.choices(
                [i+1 for i in range(len(type_ratio))], k=tot_atoms-len(type_list), weights=remain_weight)
            type_list.extend(remain_type)

        random.shuffle(type_list)
        self.atoms["type"] = type_list

        if fix_type is not None:
            self.atoms = pd.concat([self.atoms, atoms_tmp])
            self.atoms.reset_index(drop=True, inplace=True)

    def make_magmom_str(self, initial_magmom: list[float]) -> str:
        """
        VaspのMAGMOM(初期磁気モーメントを決めるパラメーター)を簡単に設定できるようにします。
        MAGMOMの仕様
        -------
            例えばpara が "Fe Co Ni"でそれらが3つずつ入った系を考えます。
            磁気モーメントをFe->4, Co->3, Ni->2 で入れたいとします。
            vaspのPOSCARにはtypeごとに座標が入ります。
            そしてMAGMOMはPOSCARの原子順に空白区切りで磁気モーメントを入れる必要があります。
            なので、例の場合は "4 4 4 3 3 3 2 2 2" というstrを返します。
        Parameter
        ---------
            initial_magmom: list[float]
                atomtypeごとの初期磁気モーメント値が入ったlistです。
                例の場合 : initial_magmom = [4,3,2] と指定します。
        Returnval
        ---------
            magmom_str: str 
                MAGMOMにこのstrを指定すればokです。
        """
        type_dict: dict[int, int] = {}
        for type_len in range(len(self.atom_symbol_to_type)):
            type_dict[type_len] = 0

        type_list = self.atoms["type"]
        for typ in type_list:
            type_dict[typ-1] += 1

        magmom_str = ""
        for type_len in range(len(self.atom_symbol_to_type)):
            for _ in range(type_dict[type_len]):
                magmom_str += str(initial_magmom[type_len])
                magmom_str += " "

        magmom_str = magmom_str[:-1]
        return magmom_str

    def change_lattice_const(self,
                             new_cell: np.array(np.float32) = None,
                             magnification: np.float32 = None):
        """
        現在の系を形はそのままに拡大（縮小）します。
        Parameters
        ----------
            new_cell: np.array(np.float32)
                変更後のセルサイズ(xyz別)
            magnification: np.float32
                セルを何倍にするか
        """
        assert new_cell or magnification, "new_cellかmagnificationのどちらかを指定してください"
        assert not (
            new_cell and magnification), "new_cellかmagnificationのどちらかを指定してください"
        assert magnification or len(new_cell) == 3, "正しい形式でnew_cellを指定してください"
        if magnification:
            new_cell = magnification*self.cell
        for i, dim in enumerate(["x", "y", "z"]):
            self.atoms[dim] *= new_cell[i]/self.cell[i]
        self.cell = new_cell

    def make_empty_space(self,
                         empty_length: float = 10.0,
                         direction: str = "z",
                         both_direction: bool = True):
        """cellに真空部分を作ります.
        Arguments
        ---------
            empty_length: float
                真空部分の長さ
            direction: str
                どの方向に真空部分を作るか("x" or "y" or "z")
            both_direction: bool
                Trueならば+方向と-方向に半分ずつスペースができる
        """
        assert direction == "x" or direction == "y" or direction == "z", "Incorrect direction"
        dim = {"x": 0, "y": 1, "z": 2}
        self.cell[dim[direction]] += empty_length
        if both_direction:
            self.atoms[direction] += empty_length / 2

    def mirroring_atoms(self, direction: str = "z"):
        """
        指定した方向に、原子を反転して結合します.

        Argument
        --------
            direction : str 
                どの方向にmirroringするか ("x" or "y" or "z")
        """
        assert direction == "x" or direction == "y" or direction == "z", "Incorrect direction"
        # make mirror sf
        sf_mirror = deepcopy(self)
        dim = {"x": 0, "y": 1, "z": 2}
        sf_mirror.atoms[direction] *= -1
        sf_mirror.atoms[direction] += sf_mirror.cell[dim[direction]]
        sf_mirror.wrap_atoms()
        # concat
        self.atoms[direction] += self.cell[dim[direction]]
        self.cell[dim[direction]] *= 2
        self.concat_atoms(sf_mirror)

    def shuffle_type_by_part(self, segment_num: list[int], type_ratio: list[int]):
        """
        xyzで区切られた領域に対して、原子タイプをシャッフルする。
        -> 大きい系で、原子タイプの比率が偏ることがないようにする。

        Arrguments
        ----------
            segment_num_num:list[int]
                [x,y,z]方向に何個領域を区切るか
            type_ratio:list[int]
                原子タイプの割合
        """
        assert len(segment_num) == 3, "segment_num is incorrrect form"
        assert len(type_ratio) == len(
            self.atom_symbol_to_type), "type_ratio is incorrect form"

        new_atoms = pd.DataFrame(columns=self.atoms.columns.to_list())

        segment_num = np.array(segment_num, dtype=np.int32)
        seg_length = self.cell / segment_num

        which_segment = np.array([0 for _ in range(len(self))])
        which_segment += self.atoms["x"] // seg_length[0]
        which_segment += self.atoms["y"] // seg_length[1] * segment_num[0]
        which_segment += self.atoms["z"] // seg_length[2] * \
            segment_num[0] * segment_num[1]

        for segment_id in range(np.prod(segment_num)):
            shuffled_seg = deepcopy(self)
            shuffled_seg.atoms = shuffled_seg.atoms[which_segment == segment_id]
            shuffled_seg.shuffle_type(type_ratio)
            new_atoms = pd.concat([new_atoms, shuffled_seg.atoms])

        new_atoms = new_atoms.reset_index(drop=True)
        self.atoms = new_atoms

    def slide_atoms(self, slide_length: list[float], change_cellsize: bool = True):
        """
        xyz方向それぞれに原子を平行移動させる。

        Arguments
        ------------------
            slide_length: list[float]
                [x, y, z] 方向にどれだけスライドさせるか
            change_cellsize: bool
                スライドに伴い,cellサイズを変更するか
                ex. x方向に 1 だけ移動-> x方向 の cellsize を1増加させる.
        """
        assert len(
            slide_length) == 3, "Specify the slide_length for the [x, y, z] direction"

        self.atoms["x"] += slide_length[0]
        self.atoms["y"] += slide_length[1]
        self.atoms["z"] += slide_length[2]

        if change_cellsize:
            self.cell += np.array(slide_length)

    def make_magmom_antimagnetic_body_str(self, initial_magmom: list[float], 
                                          magnetic_atom_type: list[int] =[],
                                          nearest_neighbor_distance: list[list[float]]=[[]]) -> str:
        """
        反強磁性体のvaspのMAGMOM(初期磁気モーメントを決めるパラメーター)を設定できます。
        MAGMOMの仕様
        -------
            例えばparaが"O Fe"で酸素が6つ鉄が4つ入った系を考えます。
            Fe2O3は反強磁性体であるため "0 0 0 0 0 0 -3.8 3.8 -3.8 3.8"のように
            酸素はmagmomが0, 鉄はmagomomの大きさが3.8で隣り合う酸素とは正負を逆に取る必要性があります。
            ちなみに鉄のmagmomを同じ符号で統一すると強磁性体になるため誤った計算結果を返すことが予想されます。
        Parameter
        ---------
            initial_magmom: list[float]
                atomtypeごとの初期磁気モーメントの大きさが入ったlistです。

            magnetic_atom_type: list[int]
                反磁性体のatomtypeが入ったlistです。
                この配列に格納されたatomtypeのmagmomは最近接反磁性原子間での値の正負が逆転する。
                注意: 指定できる原子種は1種類だけです。
            
            nearest_neighbor_distance : list[float]
                magnetic_atom_typeで指定したatomtypeの第一近接距離が入ったlistです。
                ある原子のmagmomを負に指定し、その第一近接原子のmagmomを正に指定するため必要です。

                例 : "C H O P S Zn Fe"がparaに設定されている場合
                          initial_magmom = [0, 0, 0, 0, 0, 0, 3.8] 
                          magnetic_atom_type = [7] 
                          nearest_neighbor_distance = [
                                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.5], 
                                    ]                                           
                        と指定します。
            nearest_neighbor_distance[6][6]に格納されている値が鉄原子の第一近接距離です。
            鉄-鉄間の原子以外は適当な値を入れておいてください。
            
        Returnval
        ---------
            magmom_str: str 
                MAGMOMにこのstrを指定すればokです。
        """
        assert(len(magnetic_atom_type) > 0)
        magmom_dict = np.zeros(len(self.atoms))
        for atomtype in magnetic_atom_type:
            neighbor_list = self.get_neighbor_list(mode="bond_length", bond_length=nearest_neighbor_distance)
            first_atom_num = self.atoms.loc[self.atoms["type"] == atomtype].index[0]
            magmom_dict[first_atom_num] = initial_magmom[atomtype-1]
            initial_magmom[atomtype-1] *= -1

            #1つ目の対象の原子の再隣接原子をqueueに入れる
            atoms_queue = []
            for queue_neighbor_num in neighbor_list[first_atom_num]:
                if queue_neighbor_num in self.atoms.loc[self.atoms["type"] == atomtype].index:
                    if queue_neighbor_num not in atoms_queue:
                        atoms_queue.append(queue_neighbor_num)
            atoms_queue.append(-1)
            #queueで隣の原子に符号が逆のmagmomを与える
            loop_num = 0
            while len(atoms_queue) > 0 and atoms_queue!= [-1]:
                if atoms_queue[0] >= 0:
                    if magmom_dict[atoms_queue[0]] == 0:
                        magmom_dict[atoms_queue[0]] = initial_magmom[atomtype-1]
                        #最近接同一原子かつmagmomが与えられていない原子の番号をqueueに入れる
                        for queue_neighbor_num in neighbor_list[atoms_queue[0]]:
                            if queue_neighbor_num in self.atoms.loc[self.atoms["type"] == atomtype].index:
                                if magmom_dict[queue_neighbor_num] == 0 and queue_neighbor_num not in atoms_queue:
                                    atoms_queue.append(queue_neighbor_num)
                        atoms_queue.pop(0)
                else:
                    initial_magmom[atomtype-1] *= -1
                    atoms_queue.pop(0)
                    atoms_queue.append(-1)
                if loop_num > 10*30:
                    print("ERROR: Too much loop")
                    break
                loop_num += 1
        assert (len(self.atoms.loc[self.atoms["type"] == atomtype].index) == len(list(filter(lambda x: x != 0, magmom_dict)))) 
        #self.atoms = self.atoms.assign(magmom=magmom_dict)
        
        magmom_str = ""
        for magmom_dict_element in magmom_dict:
            magmom_str += str(magmom_dict_element)
            magmom_str += " "

        magmom_str = magmom_str[:-1]
        return magmom_str
