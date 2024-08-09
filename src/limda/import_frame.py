import pandas as pd
import numpy as np
from typing import Union, Any
import pathlib
import re
import sys
import yaml
import limda.const as C


class ImportFrame(

):
    """シミュレーションしたデータを読み込むためのクラス
    一つのフレームを読み込む
    """
    atoms: pd.DataFrame
    cell: np.array  # shape:[3]
    atom_symbol_to_type: dict[str, int]
    atom_type_to_symbol: dict[int, str]
    atom_type_to_mass: dict[int, float]
    step_num: int
    limda_default: dict[str, Any]

    def __init__(self):
        pass

    def import_limda_default(self):
        """limdaのデフォルトファイル(.limda.yaml)を読み込む
        """
        limda_dot_path = pathlib.Path.home() / ".limda.yaml"
        if pathlib.Path.exists(limda_dot_path):
            with open(limda_dot_path, "r") as f:
                self.limda_default = yaml.safe_load(f)
        else:
            self.limda_default = {}

    def import_input(self, file_path: Union[str, pathlib.Path]) -> None:
        """Laichのinputファイルを読み込み、
        sf.cell, sf.atomsを更新する
        Parameters
        ----------
            file_path: Union[str, Path]
                input.rdのパス
        Note
        ----
            #fix, #move, #press, #strain, #connect, #thermofree
            #wall, #molecule,は読み込まれません
        """
        self.cell = np.array([0.0, 0.0, 0.0])

        if self.atom_symbol_to_type is None:
            raise RuntimeError("Import para first")

        with open(file_path, 'r') as ifp:
            lines = ifp.readlines()

        for idx, line in enumerate(lines):
            spline = line.split()
            if len(spline) == 0:
                continue
            if spline[0] == "#cellx":
                self.cell[0] = float(spline[2])

            if spline[0] == "#celly":
                self.cell[1] = float(spline[2])

            if spline[0] == "#cellz":
                self.cell[2] = float(spline[2])

            if spline[0] == "#masses":
                elem_num = int(spline[1])
                for _line in lines[idx+1:idx+1+elem_num]:
                    _spline = _line.split()
                    self.atom_type_to_mass[int(_spline[0])] = float(_spline[1])

            atom_data = dict()
            if spline[0] == "#atoms":
                splines = np.array([l.split()
                                   for l in lines[idx+1:idx+1+int(spline[1])]])
                # 0-indexed
                index = splines[:, 0].astype(int) - 1
                atom_data['type'] = splines[:, 1].astype(int)
                atom_data['mask'] = splines[:, 2].astype(int)
                atom_data['x'] = splines[:, 3].astype(float)
                atom_data['y'] = splines[:, 4].astype(float)
                atom_data['z'] = splines[:, 5].astype(float)
                try:
                    atom_data['vx'] = splines[:, 6].astype(float)
                    atom_data['vy'] = splines[:, 7].astype(float)
                    atom_data['vz'] = splines[:, 8].astype(float)
                except:
                    pass

                self.atoms = pd.DataFrame(data=atom_data, index=index)

    def import_para_from_list(self, atom_symbol_list: list[str]):
        """原子のリストからatom_symbol_to_type, atom_type_to_symbol, atom_type_to_massを作成する.
        Parameters
        ----------
            atom_symbol_list : list
                原子のリスト

        Example
        -------
            atom_symbol_list = ['C', 'H', 'O', 'N']
            の場合、Cの原子のタイプが1, Hの原子のタイプが2, Oの原子のタイプが3, Nの原子のタイプが4となる

        """
        if len(atom_symbol_list) == 0 and "para" in self.limda_default:
            atom_symbol_list = self.limda_default["para"]
        if len(atom_symbol_list) == 0:
            return

        atom_symbol_to_type = {}
        type_list = [i for i in range(1, len(atom_symbol_list)+1)]
        atom_symbol_to_type = {key: val for key,
                               val in zip(atom_symbol_list, type_list)}
        # type -> symbol# symbol -> type # type -> mass
        self.atom_symbol_to_type = atom_symbol_to_type
        self.atom_type_to_symbol = {
            atom_type: atom_symbol for atom_symbol, atom_type in self.atom_symbol_to_type.items()}

        self.atom_type_to_mass = {}
        for atom_symbol, atom_type in self.atom_symbol_to_type.items():
            self.atom_type_to_mass[atom_type] = C.ATOM_SYMBOL_TO_MASS[atom_symbol]

    def import_para_from_str(self, atom_symbol_str: str):
        """
            受け取ったstrをlistにして、
            import_para_from_list()を呼び出す。
            Parameters
            ----------
                para_atom_symbol_list : list   
                    空白区切りの原子の文字列

            Example
            -------
                para_atom_symbol_str = 'C H O N' #原子と原子の間には、スペース
                の場合、Cの原子のタイプが1, Hの原子のタイプが2, Oの原子のタイプが3, Nの原子のタイプが4となる
        """
        self.import_para_from_list(atom_symbol_str.split())

    def import_car(self, file_path: Union[str, pathlib.Path]) -> None:
        """ Car file を読み込み 
            self.atoms, self.cell の更新
            Parameter
            ----------
                file_path: Union[str, Path]
                    carfileのpath
        """
        input_cell = False  # car fileがcellの情報を含んでいるか
        current_row = 0
        with open(file_path, 'r') as f:
            while True:
                spline = f.readline().split()
                current_row += 1
                if len(spline) == 0:
                    continue
                if spline[0] == "PBC=ON":  # 周期境界がある
                    input_cell = True
                if spline[0] == "!DATE":
                    break
            if input_cell:
                spline = f.readline().split()
                self.cell = np.float_(spline[1:4])
                current_row += 1
        car_df = pd.read_csv(file_path,
                             names=['symbol+id', 'x', 'y', 'z',
                                    'XXXX', '1', 'xx', 'symbol', '0.000'],
                             usecols=['x', 'y', 'z', "symbol"],
                             skiprows=current_row,
                             sep="\s+")
        car_df = car_df.dropna()
        car_df.insert(0, 'type', car_df['symbol'].map(
            self.atom_symbol_to_type))  # type列を作成
        car_df.index += 1
        self.atoms = car_df[['type', 'x', 'y', 'z']]

    def import_dumppos(self, file_path: Union[str, pathlib.Path]) -> None:
        """ dumppos file を読み込み 
            self.atomsの更新
             Parameter
            ----------
                file_path: Union[str, Path]
                dumpposfileのpath
        """
        current_row = 0
        with open(file_path, 'r') as ifp:
            while True:
                current_row += 1
                spline = ifp.readline().split()
                if len(spline) == 0:
                    continue
                if spline[0] == "ITEM:" and spline[1] == "BOX":
                    slide_cell_length = np.array([None, None, None])
                    self.cell = np.array([None, None, None])
                    for dim in range(3):
                        spline = ifp.readline().split()
                        slide_cell_length[dim] = np.float64(spline[0])
                        self.cell[dim] = np.float64(spline[1])
                    current_row += 3
                    continue
                if spline[0] == "ITEM:" and spline[1] == 'ATOMS':
                    columns = spline[3:]
                    break

        self.atoms = pd.read_csv(
            file_path, skiprows=current_row, sep='\s+', names=columns)
        if 'type' in self.atoms:
            self.atoms['type'] = self.atoms['type'].astype(int)
        if 'mask' in self.atoms:
            self.atoms['mask'] = self.atoms['mask'].astype(int)

        self.atoms.index = self.atoms.index - 1
        self.atoms.sort_index(inplace=True)

        self.slide_atoms(-1 * slide_cell_length)

    def import_mol(self, molecular_fomula: str):
        '''分子式から原子配置を読み込む
        Parameters
        ----------
            molecular_fomula : str
                読み込む分子式
        '''
        from ase import build

        ase_atoms = build.molecule(molecular_fomula)
        self.atoms = pd.DataFrame(ase_atoms.positions, columns=['x', 'y', 'z'])
        self.atoms['type'] = ase_atoms.get_chemical_symbols()
        self.atoms['type'] = self.atoms['type'].map(self.atom_symbol_to_type)
        self.atoms['x'] += abs(self.atoms['x'].min()) + 0.1  # ?
        self.atoms['y'] += abs(self.atoms['y'].min()) + 0.1
        self.atoms['z'] += abs(self.atoms['z'].min()) + 0.1

    def import_vasp_poscar(self, poscar_path: Union[str, pathlib.Path]):
        """vaspに用いるPOSCARから,  
        原子それぞれの種類を表すリストを作成する。
        また、初期構造のSimulationFrame(原子の座標のみ)が得られる。
        Parameters
        ----------
            poscar_path: Union[str, Path]
                vaspで計算したディレクトリ内のPOSCARのpath         
        Note
        ----
            frameのatoms["type"]は原子の種類をtype listと照らし合した時の整数が入っています。
            速度や力がたとえ入っていたとしても、その情報は抜け落ちます。
        """
        with open(poscar_path, "r") as f:
            f.readline()
            # cell
            scaling_factor = float(f.readline())
            self.cell = np.array([None, None, None])
            for dim in range(3):
                self.cell[dim] = float(f.readline().split()[
                                       dim]) * scaling_factor
            # atom type
            atom_symbol_list = list(f.readline().split())
            for atom_symbol_num in range(len(atom_symbol_list)):
                if atom_symbol_list[atom_symbol_num][0:5] == "Type_":
                    atom_symbol_list[atom_symbol_num] = self.atom_type_to_symbol[int(atom_symbol_list[atom_symbol_num][5:])]

            atom_type_counter = list(map(int, f.readline().split()))
            total_atom_num = sum(atom_type_counter)
            atom_types = []
            for atom_type_count, atom_symbol in zip(atom_type_counter, atom_symbol_list):
                for _ in range(atom_type_count):
                    atom_types.append(self.atom_symbol_to_type[atom_symbol])
            # position
            pos_type = f.readline().split()[0]
            assert pos_type == "Cartesian" or pos_type == "Direct"
            self.atoms = pd.read_csv(
                f, sep='\s+', names=("x", "y", "z"), nrows=total_atom_num)

            if pos_type == "Direct":
                self.atoms["x"] = self.atoms["x"] * self.cell[0]
                self.atoms["y"] = self.atoms["y"] * self.cell[1]
                self.atoms["z"] = self.atoms["z"] * self.cell[2]

            self.atoms["type"] = np.array(atom_types)

    def import_xyz(self, ifn: Union[str, pathlib.Path]) -> None:
        """xyz fileを読み込む.
        Parameter
        ---------
        ifn: Union[str,Path]
            読み込むfile名
        """
        with open(ifn, 'r') as ifp:
            lines = ifp.readlines()
        total_atom = int(lines[0])
        lattice_value = re.search('Lattice="(.*?)"', lines[1])
        if lattice_value is not None:
            cellsize = lattice_value.group(1).split()
            self.cell = [0,0,0]
            if len(cellsize) == 9:
                self.cell[0] = float(cellsize[0])
                self.cell[1] = float(cellsize[4])
                self.cell[2] = float(cellsize[8])
                for cell_array in range(9):
                    if cell_array % 4 != 0:
                        assert float(cellsize[cell_array]) <=  0.01, "WARNING: LIMDA DOES NOT SUPPORT NON-RECTANGULAR."
                        
            if len(cellsize) == 3:
                self.cell[0] = float(cellsize[0])
                self.cell[1] = float(cellsize[1])
                self.cell[2] = float(cellsize[2])

        splines = np.array([l.split() for l in lines[2:2+total_atom]])
        atom_data = dict()
        try:
            atom_data['type'] = splines[:, 0].astype(int)
        except:
            atom_symbols = splines[:, 0].astype(str)
            if self.atom_symbol_to_type is None:
                print("error : atom_symbol_to_type is not defined")
                print("Import para first")
                sys.exit(-1)
            atom_data['type'] = np.array(
                [self.atom_symbol_to_type[atom_symbol]
                    for atom_symbol in atom_symbols]
            )

        for idx, dim in enumerate(['x', 'y', 'z']):
            atom_data[dim] = splines[:, idx+1].astype(float)

        index = np.arange(total_atom)
        self.atoms = pd.DataFrame(data=atom_data, index=index)
        self.atoms.sort_index(inplace=True)

    def import_cif(self, cif_file_path: Union[str, pathlib.Path]):
        """
        cif fileを読み込み
        cell, atoms, を更新する.

        Augument
        ---------
        cif_file_path : Union[str, pathlib.Path]
            input cif file path
        """
        from ase.io import read
        # reading cif file using ase
        cifdata = read(cif_file_path)
        # cell
        self.cell = cifdata.cell.array.diagonal().copy()
        # atoms position
        self.atoms = pd.DataFrame(
            cifdata.get_positions(), columns=["x", "y", "z"])
        # atoms type
        self.atoms["type"] = np.array(
            [self.atom_symbol_to_type[symbol] for symbol in cifdata.get_chemical_symbols()])

    def import_xsf(self, import_filename: Union[str, pathlib.Path], atom_type: int = 1):
        """
        xsf file を読み込み、cell, atoms[x,y,z]の情報を得る
        TODO : 多分全対応していないので解決する（atomskで作成したxsfは読み込める）

        Parameters
        ---------
            import_filename : 読み込むxsf file
            atom_type : 元素種  
        """
        with open(import_filename, "r") as f:
            lines = f.readlines()
            self.cell = np.array([float(lines[3].split()[0]), float(
                lines[4].split()[1]), float(lines[5].split()[2])])
        self.atoms = pd.read_csv(
            import_filename, skiprows=12, sep='\s+', usecols=[1, 2, 3], names=["x", "y", "z"])
        self.atoms["type"] = np.array([atom_type for _ in range(len(self))])

    def import_cfg(self, import_filename: Union[str, pathlib.Path]):
        """
        cfg file を読み込み、cell、atom[x,y,z,grain_id]の情報を得る
        TODO : 多分全対応していないので解決する（atomskで作成したcfgは読み込める）

        Parameters
        ---------
            import_filename : 読み込むcfg file
        """
        with open(import_filename, "r") as f:
            lines = f.readlines()
            current_line_id = 0
            for line in lines:
                current_line_id += 1
                spline = line.split()
                l = len(spline)
                if l == 5 and line[:21] == "Number of particles =":
                    total_atom_num = int(spline[4])
                elif l == 3 and spline[0] == "H0(1,1)":
                    cell_x = float(spline[2])
                elif l == 3 and spline[0] == "H0(2,2)":
                    cell_y = float(spline[2])
                elif l == 3 and spline[0] == "H0(3,3)":
                    cell_z = float(spline[2])
                elif l == 1 and spline[0] in self.atom_symbol_to_type:
                    atom_type = self.atom_symbol_to_type[spline[0]]
                    break
        self.cell = np.array([cell_x, cell_y, cell_z])
        self.atoms = pd.read_csv(
            import_filename, skiprows=current_line_id, sep='\s+', names=["x", "y", "z", "grain_id"])
        self.atoms[["x", "y", "z"]] *= self.cell
        self.atoms["type"] = np.array(
            [atom_type for _ in range(total_atom_num)])

    def import_file(self, import_filename: Union[str, pathlib.Path]):
        """
        file名から、適切な形式fileを読み込みます.
        Parameters
        ----------
        import_filename: str 
            読み込むファイル名
        """
        import_filename = pathlib.Path(import_filename)
        import_file_basename = import_filename.name

        if "input" in import_file_basename:
            self.import_input(import_filename)
        elif import_file_basename.endswith("xyz"):
            self.import_xyz(import_filename)
        elif import_file_basename.endswith("car"):
            self.import_car(import_filename)
        elif import_file_basename.endswith("cif"):
            self.import_cif(import_filename)
        elif "dump" in import_file_basename or "pos" in import_file_basename:
            self.import_dumppos(import_filename)
        elif import_file_basename.endswith("xsf"):
            self.import_xsf(import_filename)
        elif import_file_basename.endswith("cfg"):
            self.import_cfg(import_filename)
        elif import_file_basename == "POSCAR":
            self.import_vasp_poscar(import_filename)
        else:
            raise RuntimeError("適切なfile名にしてください.")
