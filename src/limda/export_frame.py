import numpy as np
import pandas as pd
import pathlib
import subprocess
from typing import Union
from datetime import datetime


class ExportFrame(

):
    """
    ファイルをexportするためのclass
    """

    def __init__(self):
        pass

    def export_vasp_poscar(
        self,
        ofn: str,
        comment: str = "",
        scaling_factor: float = 1.0,
    ) -> None:
        """vaspのincarファイルを作成する
        注意:この関数はsf.atomsの原子をtypeごとにinplaceに並べ替えます.

        Parameters
        ----------
            ofn: str
                出力先のパス
            comment: str
                POSCARの1行目に書かれるコメント
            scaling_factor: float
                VASPを参照してください. 基本1.0でok
        """
        for dim in range(3):
            if self.cell[dim] == 0:
                print(f'warning : cell[{dim}] is 0')

            if self.cell[dim] is None:
                print(f'warning : cell[{dim}] is not defined')
                print(f'warning : cell[{dim}] has been initialized to 0')
                self.cell[dim] = 0.0

        header_line = [
            f"{comment.strip()}\n",
            f"{scaling_factor}\n",
            f"{self.cell[0]:.10f} 0.0000000000 0.0000000000\n",
            f"0.0000000000 {self.cell[1]:.10f} 0.0000000000\n",
            f"0.0000000000 0.0000000000 {self.cell[2]:.10f}\n"
        ]
        atom_symbol_to_atom_counter = self.count_atom_types(res_type='dict')
        atom_symbols = []
        atom_types_counter = []
        for atom_type in range(1, len(self.atom_type_to_symbol) + 1):
            atom_symbol = self.atom_type_to_symbol[atom_type]
            if atom_symbol in atom_symbol_to_atom_counter and atom_symbol_to_atom_counter[atom_symbol] != 0:
                atom_types_counter.append(
                    atom_symbol_to_atom_counter[atom_symbol])
                atom_symbols.append(atom_symbol)

        header_line.append(" ".join(atom_symbols) + "\n")
        header_line.append(" ".join(map(str, atom_types_counter)) + "\n")
        header_line.append("Cartesian\n")

        self.atoms['symbol'] = self.atoms['type'].replace(
            self.atom_type_to_symbol)
        self.atoms = self.atoms.sort_values('type').reset_index(drop=True)
        with open(ofn, 'w') as ofp:
            ofp.writelines(header_line)

        self.atoms.to_csv(ofn, columns=['x', 'y', 'z'], mode='a', header=False,
                          sep=' ', float_format='%.10f', index=False)

    def export_vasp_poscar_from_contcar(
        self,
        ofn: str,
        contcar_path: str = "",
    ):
        """
        POSCARをCONTCARから作る。
        Parameters
        ----------
        ofn: str
            POSCARのPath
        contcar_path
            CONTCARのPath

        アンサンブルが異なるときは使わない方がよい
        """
        header_line = []
        with open(contcar_path, 'r') as cp:
            lines = cp.readlines()

        for line in lines:
            header_line.append(f"{line}")

        with open(ofn, 'w') as ofp:
            ofp.writelines(header_line)

    def export_vasp_incar(
            self,
            ofn: str,
            config: dict,
    ):
        """vaspのINCARファイルを作成する.
        Parameters
        ----------
            ofn: str
            出力先のファイルパス
            config: dict
            vaspの設定
            config['key'] = 'value'とすると、
            INCARでは、
            key = value
            となる
        """
        config_list = []
        for config_key, config_value in config.items():
            config_list.append(f'{config_key} = {config_value}\n')
        with open(ofn, "w") as ofp:
            ofp.writelines(config_list)

    def export_vasp_kpoints(
        self,
        ofn: str,
        comment: str = "",
        kx: int = 1,
        ky: int = 1,
        kz: int = 1,
    ):
        """vaspのKPOINTSファイルを作成する.
        Parameters
        ----------
            ofn: str
                出力先のファイルパス
            kx: int
            ky: int
            kz: int
                x, y, z方向のK点
        """
        output = [
            f"{comment.strip()}\n",
            "0\n",
            "Monkhorst\n",
            f"{kx} {ky} {kz}\n",
            "0 0 0\n"
        ]

        with open(ofn, "w") as ofp:
            ofp.writelines(output)

    def export_vasp_iconst(
        self,
        ofn: str,
        config: list[str],
    ):
        """vaspのICONSTファイルを作成する.
        Parameters
        ----------
            ofn: str
                出力先のファイルパス
            config: List[str]

        Examples
        --------
            VASPでNPT計算をする時で、セルの角度を固定したい時、
            config = ['LA 1 2 0',
                      'LA 1 3 0',
                      'LA 2 3 0'
                    ]
            とすると、ICONSTには
            ```
            LA 1 2 0
            LA 1 3 0
            LA 2 3 0
            ```
            と出力される
        """
        for line_idx in range(len(config)):
            config[line_idx] = config[line_idx].rstrip() + '\n'

        with open(ofn, "w") as ofp:
            ofp.writelines(config)

    def export_vasp_potcar(
        self,
        ofn: str,
        potcar_root: str,
        pseudopot_atom: list[str] = [],
    ):
        """vaspのPOTCARファイルを作成する.
        Parameters
        ----------
            ofn: str
                出力先のファイルパス
            potcar_root: str
                potcarが入っているフォルダのパス
            pseudopot_atom: list[str]
                POTCARを作成したいPOTLIST内のpseudopotentialsファイル名
        """
        potcar_root = pathlib.Path(potcar_root)
        make_potcar_command_list = ["cat"]

        atom_symbol_to_atom_counter = self.count_atom_types(res_type='dict')
        print(atom_symbol_to_atom_counter)
        if len(pseudopot_atom) == 0:
            print(self.atom)
            for atom_type in range(1, len(self.atom_type_to_symbol) + 1):
                atom_symbol = self.atom_type_to_symbol[atom_type]
                if atom_symbol in atom_symbol_to_atom_counter and atom_symbol_to_atom_counter[atom_symbol] != 0:
                    potcar_path = potcar_root / atom_symbol / "POTCAR"
                    make_potcar_command_list.append(f"{potcar_path.resolve()}")
            make_potcar_command_list.append(f" > {ofn}")
            make_potcar_command = " ".join(make_potcar_command_list)
            print(make_potcar_command)
            
        elif len(pseudopot_atom) > 0:
            for pseudopot_atom_type in pseudopot_atom:
                potcar_path = potcar_root / pseudopot_atom_type / "POTCAR"
                make_potcar_command_list.append(f"{potcar_path.resolve()}")
            make_potcar_command_list.append(f" > {ofn}")
            make_potcar_command = " ".join(make_potcar_command_list)
            assert len(pseudopot_atom) == len(atom_symbol_to_atom_counter), "Add all kind of atoms in pseudopot_atom: sf.vasp()"
            for atom_type_num in range(len(pseudopot_atom)):
                assert pseudopot_atom[atom_type_num][0:len(list(atom_symbol_to_atom_counter.keys())[atom_type_num])] in list(atom_symbol_to_atom_counter.keys()), "Error: The order must be the same" 
        subprocess.run(make_potcar_command, shell=True)

    def export_dumppos(self, ofn: str, time_step: int = None, out_columns=None) -> None:
        """dumpposファイルを作成する。
        Parameters
        ----------
        ofn: str
            出力するdumpposファイルの名前
        time_step: int
            dumpposファイルのtimestep
        out_columns: list[str]
            dumpposファイルに出力する列の名前の入ったlist
        """
        if out_columns is None:
            out_columns = ['type', 'mask', 'x', 'y', 'z']

        if 'mask' not in self.atoms:
            print('warning : mask is not defined')
            print('warning : mask has been initialized to 0')
            self.atoms['mask'] = np.zeros(self.get_total_atoms(),
                                          dtype=int)

        if time_step is None:
            print('warning : time_step is not defined')
            print('warning : time_step has been initialized to 0')
            time_step = 0

        for dim in range(3):
            if self.cell[dim] == 0:
                print(f'warning : cell[{dim}] is 0')

            if self.cell[dim] is None:
                print(f'warning : cell[{dim}] is not defined')
                print(f'warning : cell[{dim}] has been initialized to 0')
                self.cell[dim] = 0.0

        header_line = [
            "ITEM: TIMESTEP\n",
            f"{time_step}\n",
            "ITEM: NUMBER OF ATOMS\n",
            f"{self.get_total_atoms()}\n",
            "ITEM: BOX BOUNDS pp pp pp\n",
            f"{0.0} {self.cell[0]}\n",
            f"{0.0} {self.cell[1]}\n",
            f"{0.0} {self.cell[2]}\n",
            " ".join(["ITEM: ATOMS"] + ['id'] + out_columns) + "\n"
        ]

        with open(ofn, 'w') as ofp:
            ofp.writelines(header_line)

        # 1-indexed
        self.atoms.index = self.atoms.index + 1
        self.atoms.to_csv(ofn, columns=out_columns, mode='a', header=False,
                          sep=' ', float_format='%.6f')
        # 0-indexed
        self.atoms.index = self.atoms.index - 1

    def export_input(self, ofn: Union[str, pathlib.Path] = "input.rd", mask_info: list[str] = []) -> None:
        """input.rdを作成する。
        Parameters
        ----------
        ofn: Union[str, Path]
            input fileの名前
        mask_info: list[str]
            mask変数に対して、move,pressを行いたいときに出力する情報
            lax,laichの正しい書式で行ごとに要素にしてください。
        """
        for dim in range(3):
            if self.cell[dim] == 0:
                print(f"warning : cell[{dim}] is not defined")
                print(f"warning : cell[{dim}] has been initialized to 0")
                self.cell[dim] = 0.0
        header_line = [
            f"#cellx {0.000000}  {self.cell[0]}\n",
            f"#celly {0.000000}  {self.cell[1]}\n",
            f"#cellz {0.000000}  {self.cell[2]}\n\n",
            f"#masses {len(self.atom_type_to_mass)}\n"
        ]
        for typ, mass in self.atom_type_to_mass.items():
            header_line.append(f"{typ} {mass}\n")

        if mask_info:
            header_line.append("\n")
            for info in mask_info:
                header_line.append(f"{info}\n")

        header_line.append("\n")
        header_line.append(f"#atoms {self.get_total_atoms()}\n")

        out_columns = ['type', 'mask', 'x', 'y', 'z']

        if 'mask' not in self.atoms:
            print('warning : mask is not defined.')
            print('warning : mask has been initialized to 0.')
            self.atoms['mask'] = np.zeros(self.get_total_atoms(),
                                          dtype=int)
        if 'vx' in self.atoms and 'vy' in self.atoms and 'vz' in self.atoms:
            out_columns.extend(['vx', 'vy', 'vz'])

        self.wrap_atoms()

        def make_coods_not_zero(axises: list[str]):
            for axis in axises:
                self.atoms[axis] = np.where(
                    0, 0.0001, self.atoms[axis])
        make_coods_not_zero(['x', 'y', 'z'])

        self.atoms.index = self.atoms.index + 1  # 1-indexed
        body_line = []
        for row in self.atoms[out_columns].itertuples():
            body_line.append('    '.join(map(str, row)))
            body_line.append('\n')

        with open(ofn, 'w') as ofs:
            ofs.writelines(header_line)
            ofs.writelines(body_line)

        self.atoms.index = self.atoms.index - 1

    def export_xyz(self, ofn: Union[str, pathlib.Path],
                   out_columns: list[str] = None,
                   structure_name: str = "structure") -> None:
        """xyz fileを作成する.
        Parameters
        ----------
        ofn: Union[str Path]
            xyz fileの名前
        out_columns: list[str]
            xyzに書き込む種類を指定するlist
        structure_name: str
            構造の名前
        """
        if out_columns is None:
            out_columns = ['type', 'x', 'y', 'z']
        if structure_name == "structure":
            print('warning : structure_name is not defined')
            print('warning : structure_name has been initialized to \'structure\'')

        header_line = [
            f"{self.get_total_atoms()}\n",
            f"{structure_name}\n",
        ]

        with open(ofn, 'w') as ofp:
            ofp.writelines(header_line)
        self.atoms.to_csv(ofn, columns=out_columns, sep='\t',
                          mode='a', header=False, index=False,
                          float_format='%.6f')

    def export_car(self, export_filename: str):
        """
        car fileを出力する.

        Parameter
        ------------
        export_filename
            出力するcarfileの名前
        """
        has_cell: bool = (self.cell is not None)
        now = datetime.now()
        output_date = now.strftime("%a %b %d %H:%M:%S %Y")
        header_line = [
            "!BIOSYM archive 3\n",
            "",
            "Materials Studio Generated CAR File\n",
            f"!DATE {output_date} \n"
        ]
        if has_cell:
            header_line[1] = "PBC=ON\n"
            header_line.append(
                f"PBC {self.cell[0]:8.4f} {self.cell[1]:8.4f} {self.cell[2]:8.4f}    90.0000   90.0000   90.0000 (P1)\n")
        else:
            header_line[1] = "PBC=OFF\n"
        car_df = self.atoms.sort_values(by="type")
        typ_cnt = np.array([0 for _ in range(len(self.atom_symbol_to_type))])

        def make_atom_line_for_car(atom):
            typ_cnt[int(atom['type']) - 1] += 1
            symbol = self.atom_type_to_symbol[atom['type']]
            return f"{symbol+str(typ_cnt[int(atom['type']) - 1]):5}  {atom['x']:13.9f}  {atom['y']:13.9f}  {atom['z']:13.9f}  XXXX 1      xx     {symbol:<3} 0.000\n"
        atom_lines = car_df.apply(make_atom_line_for_car, axis=1).to_numpy()
        with open(export_filename, 'w') as ofp:
            ofp.writelines(header_line)
            ofp.writelines(atom_lines)
            ofp.writelines("end\n")
            ofp.writelines("end\n")

    def export_file(self, export_filename: str):
        """引数のfile名に合った種類の形式でfileを作成.
        Parameter
        ---------
        export_filename: str 
            作成するfile名
        """
        export_filename = pathlib.Path(export_filename)
        export_file_basename = export_filename.name
        if "input" in export_file_basename:
            self.export_input(export_filename)
        elif export_file_basename.endswith('xyz'):
            self.export_xyz(export_filename)
        elif "dump" in export_file_basename or "pos" in export_file_basename:
            self.export_dumppos(export_filename)
        elif export_file_basename.endswith('car'):
            self.export_car(export_filename)
        else:
            raise RuntimeError("適切なfile名にしてください.")
