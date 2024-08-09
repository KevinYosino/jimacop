import numpy as np
import pathlib
from tqdm import tqdm
import subprocess
import os
import time
from typing import Union, Dict, Any
import torch
import shutil

try:
    from ase.build import molecule
    from ase import Atoms
except:
    pass


class Calculate(

):
    """計算を実行するクラス
    """

    def __init__(self):
        pass

    def vasp(
            self,
            calc_directory: str = "",
            system_name: str = "",
            step_num: int = 0,
            poscar_comment: str = "",
            poscar_scaling_factor: float = 1.0,
            incar_config: dict = None,
            potcar_root: str = None,
            kpoints_comment: str = "",
            kpoints_kx: int = 1,
            kpoints_ky: int = 1,
            kpoints_kz: int = 1,
            iconst_config: list[str] = None,
            vasp_command: str = "vasp_std",
            print_vasp: bool = True,
            exist_ok: bool = False,
            poscar_from_contcar: bool = False,
            contcar_path: str = "",
            place: str = "kbox",
            num_nodes: int = 1,
            pseudopot_atom: list[str] = [],
    ):
        """vaspを実行する.
        Parameters
        ----------
            calc_directory: str
                vaspで計算を実行するディレクトリ
            system_name: str
                vaspで計算を実行する系の名前
            step_num: int
                vaspで計算を実行するファイルが何個目のファイルか
            poscar_comment: str
                POSCARの1行目に書かれるコメント
            poscar_scaling_factor: float
                vaspのドキュメントをを参照してください.基本的に変える必要はない
            potcar_root: str
                元のPOTCARがあるフォルダ
            incar_config: dict
                INCARに書かれるconfig
            kpoints_comment: str
                KPOINTSの1行目に書かれるコメント
            kpoints_kx: int
            kpoints_ky: int
            kpoints_kz: int
                x, y, z方向のK点     
            iconst_config: List[str]
                ICONSTに書かれるconfig
                VASPでNPT計算をする時で、セルの角度を固定したい時は以下のように指定する
                iconst_config = ['LA 1 2 0',
                                'LA 1 3 0',
                                'LA 2 3 0']
            vasp_command: str
                実行するvaspのコマンド
            print_vasp: bool
                実行中のvaspの標準出力をpythonからも表示するかどうか
            exist_ok: bool
                calc_directoryが存在する時、exist_ok=Trueのときは上書きしてvaspを実行する
                calc_directoryが存在する時、exist_ok=Falseのときは上書きしない
            poscar_from_contcar: bool
                poscarをcontcarから作るときTrueにする
            contcar_path: str
                contcarからposcarを作るときに必要なcontcarのpath
        """
        assert potcar_root is not None, "There isn't potcar_root."
        assert incar_config is not None, "There isn't incar_config."
        assert place == "kbox" or place.upper() == "MASAMUNE", "Invalid place"
        if not calc_directory:
            calc_directory = f"{system_name}_{step_num}"
        calc_directory = pathlib.Path(calc_directory)
        for key in ["NCORE", "NPAR", "KPAR"]:
            if key not in incar_config:
                incar_config[key] = 1

        os.makedirs(calc_directory, exist_ok=exist_ok)
        if poscar_from_contcar:
            incar_config["ISTART"] = 1
            subprocess.run(
                ["cp", f"./{system_name}_{step_num-1}/WAVECAR", f'./{calc_directory}'])
        poscar_path = calc_directory / "POSCAR"
        incar_path = calc_directory / "INCAR"
        potcar_path = calc_directory / "POTCAR"
        kpoints_path = calc_directory / "KPOINTS"
        iconst_path = calc_directory / "ICONST"
        num_process = incar_config["NCORE"] * \
            incar_config["NPAR"] * incar_config["KPAR"]
        assert num_process % num_nodes == 0, "Invalid num_nodes"
        if not poscar_from_contcar:
            self.export_vasp_poscar(
                poscar_path, poscar_comment, poscar_scaling_factor)
        else:
            self.export_vasp_poscar_from_contcar(poscar_path, contcar_path)
        self.export_vasp_incar(incar_path, incar_config)
        self.export_vasp_potcar(potcar_path, potcar_root, pseudopot_atom)
        self.export_vasp_kpoints(
            kpoints_path, kpoints_comment, kpoints_kx, kpoints_ky, kpoints_kz)

        if iconst_config is not None:
            self.export_vasp_iconst(iconst_path, iconst_config)

        if place == "kbox":
            cmd = f'mpiexec -np {num_process} {vasp_command} > stdout'
        elif place.upper() == "MASAMUNE":
            cmd = f'aprun -n {num_process} -N {int(num_process / num_nodes)} -j 1 {vasp_command} > stdout'
        vasp_md_process = subprocess.Popen(cmd, cwd=calc_directory, shell=True)
        time.sleep(5)
        if print_vasp:
            tail_process = subprocess.Popen(
                f'tail -F stdout', cwd=calc_directory, shell=True)
            while vasp_md_process.poll() is None:
                time.sleep(1)
            tail_process.kill()

    def laich(self,
              calc_dir: str = 'laich_calc',
              para_file_path: str = None,
              laich_cmd: str = 'laich',
              laich_config: dict = None,
              print_laich: bool = False,
              exist_ok=False,
              place: str = "kbox",
              num_nodes: int = 1,
              mask_info: list[str] = None):
        """LaichでMD,または構造最適化を実行する。
        Parameters
        ----------
            calc_dir: str
                laichの結果が出力される部分
            para_file_path: str
                para fileのpath
            laich_cmd: str
                laichを実行するときのコマンド
            laich_config: dict
                変数の入ったdict
            print_laich: bool
                out fileをprintするか
            exist_ok: bool
        """
        calc_dir = pathlib.Path(calc_dir)
        if calc_dir.exists() and not exist_ok:
            raise RuntimeError(f"calc_dir ({calc_dir}) exists")
        calc_dir.mkdir(parents=True, exist_ok=exist_ok)

        if para_file_path is not None:
            para_file_path = pathlib.Path(para_file_path)
        input_file_path = calc_dir / 'input.rd'
        config_file_path = calc_dir / 'config.rd'
        self.export_input(input_file_path, mask_info)
        with open(config_file_path, 'w') as f:
            for key in laich_config.keys():
                f.write(f"{key} {laich_config[key]}\n")

        if para_file_path is not None:
            try:
                subprocess.run(
                    f'cp {para_file_path} {calc_dir / "para.rd"}', shell=True)
            except:
                pass

        num_process = laich_config["MPIGridX"] * \
            laich_config["MPIGridY"]*laich_config["MPIGridZ"]
        assert num_process % num_nodes == 0, "Invalid num_nodes"

        if place == "kbox":
            cmd = f"mpiexec.hydra -np {num_process} {laich_cmd} < /dev/null >& out"
        elif place.upper() == "MASAMUNE":
            cmd = f'aprun -n {num_process} -N {int(num_process / num_nodes)} -j 1 {laich_cmd} > stdout'

        laich_process = subprocess.Popen(cmd, cwd=calc_dir, shell=True)
        time.sleep(5)
        if print_laich:
            tail_process = subprocess.Popen(
                f'tail -F out', cwd=calc_dir, shell=True)
        while laich_process.poll() is None:
            time.sleep(1)
        if print_laich:
            tail_process.kill()

        dumppos_paths = list(calc_dir.glob('./dump.pos.*'))
        dumppos_paths.sort(reverse=True)
        assert len(dumppos_paths) != 0, "dumpposが生成されていません"
        optimized_dumppos_path = dumppos_paths[0]
        self.import_dumppos(optimized_dumppos_path)

    def packmol(self,
                sf_list: list,
                pack_num_list: list[int],
                tolerance: float = 2.0,
                packmol_tmp_dir: Union[str, pathlib.Path] = "./packmol_tmp",
                xyz_condition: list[float] = None,
                seed: int = -1,
                packmol_cmd: str = "packmol",
                print_packmol: bool = False,
                exist_ok: bool = True
                ):
        """packmolで原子を詰める
        Parameters
        ----------
            sf_list : list[SimulationFrame]
                詰めるsfのlist
            pack_num_list : list
                詰めるsdatの個数のリスト
            tolerance : float
                最小の原子間距離, 原子を詰める時に原子間がtolerance以上になるように詰める
            xyz_condition : list
                詰める原子の座標の条件
            packmol_tmp_dir : Union[str,pathlib.Path]
                packmolを動かすときの一時的なディレクトリ
            seed : int
                シード値
                seed = -1のときはseedは時間で決定される
            packmol_cmd : str
                packmolを実行するときのコマンド
            print_packmol : bool
                packmolの標準出力を表示するか
        Example
        -------
        xyz_condition = [
          # [xmin, ymin, zmin, xmax, ymax, zmax] で指定する
            [2, 2, 2, 8, 18, 28],  # sf2の条件
            [2, 2, 2, 8, 10, 10]   # sf3の条件
        ]
        sf1.packmol(sf_list=[sdat2, sdat3], pack_num_list=[5, 8], xyz_condition=xyz_condition)
        とすると, sf1にsf2の原子が5個, sf3の原子が8個詰められる
        sf2は2 <= x <= 8 かつ 2 <= y <= 18 かつ 2 <= z <= 28 の位置のみに詰められる
        sdat3は2 <= x <= 8 かつ 2 <= y <= 10 かつ 2 <= z <= 10 の位置のみに詰められる

        Note
        ----
            周期境界条件の場合は端まで詰めると計算が回らなくなるので注意.
            境界には間を開けることを推奨
        """
        if xyz_condition is not None:
            assert len(xyz_condition) == len(
                sf_list), "sf_listとxyz_conditonを同じ長さにしてください."

        packmol_tmp_dir = pathlib.Path(packmol_tmp_dir)
        if packmol_tmp_dir.exists() and not exist_ok:
            raise RuntimeError(f"packmol_tmp_dir ({packmol_tmp_dir}) exists")
        packmol_tmp_dir.mkdir(parents=True, exist_ok=exist_ok)

        first_line = [
            f"tolerance {tolerance}\n",
            f"filetype xyz\n",
            f"seed {seed}\n",
            f"output packmol_mixture_result.xyz\n\n"
        ]

        second_line = []
        if self.atoms is not None and self.get_total_atoms() >= 1:
            second_line = [
                "structure this_sf.xyz\n",
                "\tnumber 1\n",
                "\tfixed 0. 0. 0. 0. 0. 0. 0. \n",
                "end structure\n\n",
            ]

        with open(packmol_tmp_dir / "packmol_mixture_comment.inp", 'w') as f:
            f.writelines(first_line)
            f.writelines(second_line)
            for frame_idx in range(len(sf_list)):
                f.write(f"structure sf_idx_{frame_idx}.xyz\n")
                f.write(f"\tnumber {pack_num_list[frame_idx]}\n")
                if xyz_condition is not None:
                    f.write(f"\tinside box")
                    for condition in xyz_condition[frame_idx]:
                        f.write(f"\t{condition}")
                    f.write("\n")
                f.write("end structure\n\n")

        # export xyz file
        if self.atoms is not None and self.get_total_atoms() >= 1:
            self.export_xyz(packmol_tmp_dir / "this_sf.xyz",
                            structure_name="this_sf")

        for frame_idx in range(len(sf_list)):
            sf_list[frame_idx].export_xyz(packmol_tmp_dir / f"sf_idx_{frame_idx}.xyz",
                                          structure_name=f"sf_idx_{frame_idx}")

        # run packmol
        if print_packmol:
            p = subprocess.Popen(
                f"{packmol_cmd} < packmol_mixture_comment.inp", cwd=packmol_tmp_dir, shell=True, )
        else:
            p = subprocess.Popen(f"{packmol_cmd} < packmol_mixture_comment.inp", cwd=packmol_tmp_dir, shell=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        p.wait()
        # import result
        self.import_xyz(packmol_tmp_dir / "packmol_mixture_result.xyz")

    def lax(self,
            calc_dir: str = "lax_calc",
            lax_cmd: str = "lax",
            lax_config: dict = None,
            print_lax: bool = False,
            exist_ok=False,
            place: str = "kbox",
            num_nodes: int = 1,
            mask_info: list[str] = [],
            omp_num_threads: int = 1):
        """
        laxを実行する.
        Parameters
        ----------
        calc_dir
            計算が行なわれるディレクトリ、ここにdumpposが出力されます。
        lax_cmd
            laxの実行ファイルのpath
        lax_config
            laxのconfig fileに書き出される変数のdict
        print_lax
            out fileを出力するか
        exist_ok
            calc_dirが存在するときに計算を行うか
        mask_info
            input fileにそのまま出力されるmoveやpressの情報
        omp_num_threads
            OMP_NUM_THREADSの値
        """
        calc_dir = pathlib.Path(calc_dir)
        if calc_dir.exists() and not exist_ok:
            raise RuntimeError(f"calc_dir ({calc_dir}) exists")
        calc_dir.mkdir(parents=True, exist_ok=exist_ok)
        # input
        input_file_path: pathlib.Path = calc_dir / "input.rd"
        self.export_input(input_file_path, mask_info)
        # config
        config_file_path: pathlib.Path = calc_dir / "config.rd"
        with open(config_file_path, "w") as f:
            for key in lax_config.keys():
                f.write(f"{key} {lax_config[key]}\n")
        if "MPIGridX" not in lax_config:
            lax_config["MPIGridX"] = 1
        if "MPIGridY" not in lax_config:
            lax_config["MPIGridY"] = 1
        if "MPIGridZ" not in lax_config:
            lax_config["MPIGridZ"] = 1
        num_process = int(
            lax_config["MPIGridX"]) * int(lax_config["MPIGridY"]) * int(lax_config["MPIGridZ"])
        assert num_process % num_nodes == 0, "Invalid num_nodes"

        assert omp_num_threads >= 1, "omp_num_threads must be an integer greater than or equal to 1"
        if omp_num_threads > 1:
            os.environ["OMP_NUM_THREADS"] = f"{omp_num_threads}"

        if place == "kbox":
            cmd = f"mpiexec.hydra -np {num_process} {lax_cmd} < /dev/null >& out"
        elif place.upper() == "MASAMUNE":
            cmd = f'aprun -n {num_process} -N {int(num_process / num_nodes)} -j 1 {lax_cmd} > out'

        lax_process = subprocess.Popen(cmd, cwd=calc_dir, shell=True)
        time.sleep(5)
        if print_lax:
            tail_process = subprocess.Popen(
                f"tail -F out", cwd=calc_dir, shell=True)
        while lax_process.poll() is None:
            time.sleep(1)
        if print_lax:
            tail_process.kill()
        dumppos_paths = list(calc_dir.glob("./dump.pos.*"))
        dumppos_paths.sort(reverse=True)
        assert len(dumppos_paths) != 0, "dumpposが生成されていません"
        optimized_dumppos_path = dumppos_paths[0]
        self.import_dumppos(optimized_dumppos_path)

        if omp_num_threads > 1:
            os.environ["OMP_NUM_THREADS"] = "1"

    def allegro(self,
                cut_off: float,
                device: Union[str, torch.DeviceObjType],
                allegro_model: torch.jit._script.RecursiveScriptModule,
                flag_calc_virial=False,
                ) -> Dict[str, torch.Tensor]:
        """Allegroを使って、sfに入っている原子の座標に対して推論を行い、
        ポテンシャルエネルギーと原子に働く力を計算する
        Allegroによって予測されたポテンシャルエネルギーはsf.pred_potential_energyに入る
        Allegroによって予測されたそれぞれの原子にかかる力はsf.atoms.loc[:, ["pred_fx", "pred_fy", "pred_fz"]]に入る
        Parameters
        ----------
            cut_off: float
                原子の相互作用のカットオフ半径
            device: Union[str, torch.DeviceObjType]
                どのデバイスでAllegroの計算を行うか, "cpu" or "cuda"
            allegro_model: torch.jit._script.RecursiveScriptModule
                frozenされたAllegroを読み込んだモデル
                pathではないことに注意
        """

        if type(device) == str:
            device = torch.device(device)

        cell = np.array(self.cell, dtype=np.float32)
        pos = np.array(self.atoms[["x", "y", "z"]].values, dtype=np.float32)
        atom_types = np.array(self.atoms["type"].values)
        atom_types -= 1
        cut_off = np.array(cut_off, dtype=np.float32)

        edge_index = [[], []]
        edge_index = self.get_edge_index(cut_off=cut_off)
        edge_index = np.array(edge_index)

        pos_tensor = torch.tensor(pos, device=device)
        edge_index_tensor = torch.tensor(edge_index, device=device)
        cell_tensor = torch.tensor(cell, device=device)
        atom_types_tensor = torch.tensor(atom_types, device=device)
        cut_off_tensor = torch.tensor(cut_off, device=device)

        output = allegro_model(
            pos_tensor,
            edge_index_tensor,
            cell_tensor,
            atom_types_tensor,
            cut_off_tensor,
            flag_calc_virial,
        )

        self.atoms[['pred_fx', 'pred_fy', 'pred_fz']
                   ] = output['force'].cpu().detach().numpy()
        self.atoms['pred_potential_energy'] = output['atomic_energy'].cpu(
        ).detach().numpy()
        self.pred_potential_energy = output['total_energy'].cpu(
        ).detach().item()
        if flag_calc_virial:
            self.pred_virial_tensor = output['virial'].cpu().detach().numpy()

        return output

    def check_vasp(self,
                   incar_config: Dict[str, Any],
                   magmom_list: list[float] = None,
                   check_magmom: bool = True,
                   check_potim: bool = True,
                   light_atom_border: float = 10.0,
                   potim: tuple[float, float] = (1.0, 2.0),
                   ) -> tuple[Dict[str, Any], list[str]]:
        """
        vaspを回す(self.run_vasp)前に条件(self, incar_config)をcheckする
        MAGMOM : ISPINが有効なら、magmomlistに合わせて、MAGMOMを設定する
        POTIM : 軽元素によってPOTIMを変える
        iconst_config : NPTならばセルが傾かないように設定する

        Parameters
        ----------
            incar_config : incar_config, 
            magmom_list : magmomを設定するときに必要
            check_magmom : MAGMOMをcheckするかどうか
            check_potim : POTIMをcheckするかどうか
            light_atom_border : 軽元素とみなす最大の重さ
            potim : (軽元素があるとき、ないとき) のPOTIM
        """
        # MAGMOM
        if check_magmom:
            if incar_config["ISPIN"] == 2 and magmom_list is not None:
                incar_config["MAGMOM"] = self.make_magmom_str(magmom_list)

        # POTIM
        if check_potim:
            type_set = self.get_atom_type_set()
            lightest_mass = 1000.
            for typ in type_set:
                lightest_mass = min(lightest_mass, self.atom_type_to_mass[typ])
            if lightest_mass <= light_atom_border:  # 軽元素が含まれるか
                incar_config["POTIM"] = potim[0]
            else:
                incar_config["POTIM"] = potim[1]

        # iconst_config
        iconst_config: list[str] = None
        if incar_config["ISIF"] == 3:
            iconst_config = [  # 角度一定でNPTする設定
                'LA 1 2 0',
                'LA 1 3 0',
                'LA 2 3 0'
            ]

        return incar_config, iconst_config
