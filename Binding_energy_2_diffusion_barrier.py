import numpy as np
import os
import cv2
import pandas as pd

VIB_binding_energy_path = "C:\\Users\\Administrator\\Desktop\\软件包\\BaiduSyncdisk\\CNN-dataset-split\\binding-energy-2-diffuision barrier\\VIB-dataset\\"
IVB_VB_binding_energy_path = "C:\\Users\\Administrator\\Desktop\\软件包\\BaiduSyncdisk\\CNN-dataset-split\\binding-energy-2-diffuision barrier\\IVB-VB-dataset\\"

# VIB_Path_number
Path1_number = np.array([[6,15],[15,0],[0,12],[12,7],[7,16],[16,4],[4,19],[19,10],[10,19],[19,4],[4,16],[16,7],[7,12],[12,0],[0,15],[15,6]])
Path2_number = np.array([[6,15],[15,0],[0,12],[12,7],[7,18],[18,2],[2,18],[18,7],[7,12],[12,0],[0,15],[15,6]])
Path3_number = np.array([[6,15],[15,0],[0,12],[12,7],[7,18],[18,2],[2,14],[14,39],[39,14],[14,2],[2,18],[18,7],[7,12],[12,0],[0,15],[15,6]])
# IVB_VB_Path_number
#Path1_number = np.array([[17,20],[20,2],[2,24],[24,10],[10,18],[18,0],[0,21],[21,9],[9,21],[21,0],[0,18],[18,10],[10,24],[24,2],[2,20],[20,17]])
#Path2_number = np.array([[17,20],[20,2],[2,24],[24,10],[10,22],[22,4],[4,22],[22,10],[10,24],[24,2],[2,20],[20,17]])
#Path3_number = np.array([[17,20],[20,2],[2,24],[24,10],[10,22],[22,4],[4,25],[25,13],[13,25],[25,4],[4,22],[22,10],[10,24],[24,2],[2,20],[20,17]])

DFT_dataset = (pd.read_csv(VIB_binding_energy_path + 'final-VIB-DFT-dataset.csv', encoding='GB18030', header=0)).values
TL_dataset = (pd.read_csv(VIB_binding_energy_path + 'final_VIB_TL_dataset.csv', encoding='GB18030', header=0)).values

#DFT_dataset = (pd.read_csv(IVB_VB_binding_energy_path + 'IVB-VB-DFT-dataset.csv', encoding='GB18030', header=0)).values
#TL_dataset = (pd.read_csv(IVB_VB_binding_energy_path + 'IVB_VB_TL_dataset.csv', encoding='GB18030', header=0)).values

Reshape_DFT_dataset = DFT_dataset.reshape(36, 40, 2)
Reshape_TL_dataset = TL_dataset.reshape(36, 40, 2)

#Reshape_DFT_dataset = DFT_dataset.reshape(70, 45, 2)
#Reshape_TL_dataset = TL_dataset.reshape(70, 45, 2)

DFT_dataset_0 = Reshape_DFT_dataset[0,:,:]

surface = ['HfS2', 'HfSe2', 'NbS2', 'TaS2', 'TiS2', 'VS2', 'VSe2', 'ZrS2']
VIB_surface = ['MoS2', 'MoSe2', 'WS2', 'WSe2']
doped_atom = ['Al', 'Co', 'Cr', 'Cu', 'Fe', 'Mn', 'Ni', 'V', 'Zn']

def E_btoE_d(Path_number,E_b_dataset):
    path_diffusion_barrier = []
    for i in range(len(Path_number)):
        for j in range(len(E_b_dataset)):
            if E_b_dataset[j,1] == Path_number[i,0]:
                A = E_b_dataset[j,0]
            if E_b_dataset[j,1] == Path_number[i,1]:
                B = E_b_dataset[j,0]
        path_diffusion_barrier.append(A-B)
    return path_diffusion_barrier

def E_dtoProfile(path_barrier):
    path_profile = [0]
    for i in range(len(path_barrier)):
        sum = path_profile[i] + path_barrier[i]
        path_profile.append(sum)
    return path_profile

def profile_peak(energy_profile):
    profile_max = 0
    for i in range(len(energy_profile)):
        if profile_max < energy_profile[i]:
            profile_max = energy_profile[i]
    return profile_max

num = 0

Total_profile_peak_DFT = os.path.join(VIB_binding_energy_path, 'total_profile_peak_DFT.txt')
Total_profile_peak_TL = os.path.join(VIB_binding_energy_path, 'total_profile_peak_TL.txt')

with open(Total_profile_peak_DFT, 'a') as f:
    f.write('Total_profile_peak_DFT')
    f.write("\n")

with open(Total_profile_peak_TL, 'a') as f:
    f.write('Total_profile_peak_TL')
    f.write("\n")

for m in range(len(doped_atom)):
    for n in range(len(VIB_surface)):
        doped_surface = VIB_surface[n] + '_' + doped_atom[m]
        DFT_dataset_sub = Reshape_DFT_dataset[num, :, :]
        TL_dataset_sub = Reshape_TL_dataset[num, :, :]
        num += 1
        # Find the continuous diffusion barriers for the three paths
        Path1_barrier_DFT = E_btoE_d(Path1_number, DFT_dataset_sub)
        Path1_barrier_TL = E_btoE_d(Path1_number, TL_dataset_sub)
        Path2_barrier_DFT = E_btoE_d(Path2_number, DFT_dataset_sub)
        Path2_barrier_TL = E_btoE_d(Path2_number, TL_dataset_sub)
        Path3_barrier_DFT = E_btoE_d(Path3_number, DFT_dataset_sub)
        Path3_barrier_TL = E_btoE_d(Path3_number, TL_dataset_sub)
        # Find the energy curves of the three paths
        Path1_profile_DFT = E_dtoProfile(Path1_barrier_DFT)
        Path1_profile_TL = E_dtoProfile(Path1_barrier_TL)
        Path2_profile_DFT = E_dtoProfile(Path2_barrier_DFT)
        Path2_profile_TL = E_dtoProfile(Path2_barrier_TL)
        Path3_profile_DFT = E_dtoProfile(Path3_barrier_DFT)
        Path3_profile_TL = E_dtoProfile(Path3_barrier_TL)
        # Find the path peak
        Path1_profile_max_DFT = profile_peak(Path1_profile_DFT)
        Path1_profile_max_TL = profile_peak(Path1_profile_TL)
        Path2_profile_max_DFT = profile_peak(Path2_profile_DFT)
        Path2_profile_max_TL = profile_peak(Path2_profile_TL)
        Path3_profile_max_DFT = profile_peak(Path3_profile_DFT)
        Path3_profile_max_TL = profile_peak(Path3_profile_TL)

        Path1_name_DFT = doped_surface + '_Path1_DFT'
        Path2_name_DFT = doped_surface + '_Path2_DFT'
        Path3_name_DFT = doped_surface + '_Path3_DFT'
        Path1_name_TL = doped_surface + '_Path1_TL'
        Path2_name_TL = doped_surface + '_Path2_TL'
        Path3_name_TL = doped_surface + '_Path3_TL'

        Path1_profile_outfile_DFT = os.path.join(VIB_binding_energy_path, Path1_name_DFT + '.txt')
        Path2_profile_outfile_DFT = os.path.join(VIB_binding_energy_path, Path2_name_DFT + '.txt')
        Path3_profile_outfile_DFT = os.path.join(VIB_binding_energy_path, Path3_name_DFT + '.txt')
        Path1_profile_outfile_TL = os.path.join(VIB_binding_energy_path, Path1_name_TL + '.txt')
        Path2_profile_outfile_TL = os.path.join(VIB_binding_energy_path, Path2_name_TL + '.txt')
        Path3_profile_outfile_TL = os.path.join(VIB_binding_energy_path, Path3_name_TL + '.txt')

        with open(Total_profile_peak_DFT, 'a') as f:
            f.write(str(Path1_profile_max_DFT))
            f.write("\n")
            f.write(str(Path2_profile_max_DFT))
            f.write("\n")
            f.write(str(Path3_profile_max_DFT))
            f.write("\n")

        with open(Total_profile_peak_TL, 'a') as f:
            f.write(str(Path1_profile_max_TL))
            f.write("\n")
            f.write(str(Path2_profile_max_TL))
            f.write("\n")
            f.write(str(Path3_profile_max_TL))
            f.write("\n")

"""
# IVB_VB

Total_barrier_DFT = os.path.join(IVB_VB_binding_energy_path, 'total_barrier_DFT.txt')
Total_barrier_TL = os.path.join(IVB_VB_binding_energy_path, 'total_barrier_TL.txt')
Total_profile_peak_DFT = os.path.join(IVB_VB_binding_energy_path, 'total_profile_peak_DFT.txt')
Total_profile_peak_TL = os.path.join(IVB_VB_binding_energy_path, 'total_profile_peak_TL.txt')

# with open(Total_barrier_DFT, 'a') as f:
#     f.write('Total_barrier_DFT')
#     f.write("\n")
# with open(Total_barrier_TL, 'a') as f:
#     f.write('Total_barrier_TL')
#     f.write("\n")

with open(Total_profile_peak_DFT, 'a') as f:
    f.write('Total_profile_peak_DFT')
    f.write("\n")

with open(Total_profile_peak_TL, 'a') as f:
    f.write('Total_profile_peak_TL')
    f.write("\n")

for m in range(len(doped_atom)):
    if doped_atom[m] == 'V':
        for n in range(len(surface)):
            if surface[n] == 'VS2' or surface[n] == 'VSe2':
                #num += 1
                continue
            else:
                doped_surface = surface[n] + '_' + doped_atom[m]
                DFT_dataset_sub = Reshape_DFT_dataset[num, :, :]
                TL_dataset_sub = Reshape_TL_dataset[num, :, :]
                num += 1
                # Find the continuous diffusion barriers for the three paths
                Path1_barrier_DFT = E_btoE_d(Path1_number, DFT_dataset_sub)
                Path1_barrier_TL = E_btoE_d(Path1_number, TL_dataset_sub)
                Path2_barrier_DFT = E_btoE_d(Path2_number, DFT_dataset_sub)
                Path2_barrier_TL = E_btoE_d(Path2_number, TL_dataset_sub)
                Path3_barrier_DFT = E_btoE_d(Path3_number, DFT_dataset_sub)
                Path3_barrier_TL = E_btoE_d(Path3_number, TL_dataset_sub)
                # Find the energy curves of the three paths
                Path1_profile_DFT = E_dtoProfile(Path1_barrier_DFT)
                Path1_profile_TL = E_dtoProfile(Path1_barrier_TL)
                Path2_profile_DFT = E_dtoProfile(Path2_barrier_DFT)
                Path2_profile_TL = E_dtoProfile(Path2_barrier_TL)
                Path3_profile_DFT = E_dtoProfile(Path3_barrier_DFT)
                Path3_profile_TL = E_dtoProfile(Path3_barrier_TL)
                # Find the path peak
                Path1_profile_max_DFT = profile_peak(Path1_profile_DFT)
                Path1_profile_max_TL = profile_peak(Path1_profile_TL)
                Path2_profile_max_DFT = profile_peak(Path2_profile_DFT)
                Path2_profile_max_TL = profile_peak(Path2_profile_TL)
                Path3_profile_max_DFT = profile_peak(Path3_profile_DFT)
                Path3_profile_max_TL = profile_peak(Path3_profile_TL)

                Path1_name_DFT = doped_surface + '_Path1_DFT'
                Path2_name_DFT = doped_surface + '_Path2_DFT'
                Path3_name_DFT = doped_surface + '_Path3_DFT'
                Path1_name_TL = doped_surface + '_Path1_TL'
                Path2_name_TL = doped_surface + '_Path2_TL'
                Path3_name_TL = doped_surface + '_Path3_TL'

                Path1_profile_outfile_DFT = os.path.join(IVB_VB_binding_energy_path, Path1_name_DFT + '.txt')
                Path2_profile_outfile_DFT = os.path.join(IVB_VB_binding_energy_path, Path2_name_DFT + '.txt')
                Path3_profile_outfile_DFT = os.path.join(IVB_VB_binding_energy_path, Path3_name_DFT + '.txt')
                Path1_profile_outfile_TL = os.path.join(IVB_VB_binding_energy_path, Path1_name_TL + '.txt')
                Path2_profile_outfile_TL = os.path.join(IVB_VB_binding_energy_path, Path2_name_TL + '.txt')
                Path3_profile_outfile_TL = os.path.join(IVB_VB_binding_energy_path, Path3_name_TL + '.txt')
                Total_barrier_DFT = os.path.join(IVB_VB_binding_energy_path, 'total_barrier_DFT.txt')
                Total_barrier_TL = os.path.join(IVB_VB_binding_energy_path, 'total_barrier_TL.txt')


                with open(Total_profile_peak_DFT,'a') as f:
                    f.write(str(Path1_profile_max_DFT))
                    f.write("\n")
                    f.write(str(Path2_profile_max_DFT))
                    f.write("\n")
                    f.write(str(Path3_profile_max_DFT))
                    f.write("\n")

                with open(Total_profile_peak_TL,'a') as f:
                    f.write(str(Path1_profile_max_TL))
                    f.write("\n")
                    f.write(str(Path2_profile_max_TL))
                    f.write("\n")
                    f.write(str(Path3_profile_max_TL))
                    f.write("\n")
    else:
        for n in range(len(surface)):
            doped_surface = surface[n] + '_' + doped_atom[m]
            DFT_dataset_sub = Reshape_DFT_dataset[num, :, :]
            TL_dataset_sub = Reshape_TL_dataset[num, :, :]
            num += 1
            # Find the continuous diffusion barriers for the three paths
            Path1_barrier_DFT = E_btoE_d(Path1_number, DFT_dataset_sub)
            Path1_barrier_TL = E_btoE_d(Path1_number, TL_dataset_sub)
            Path2_barrier_DFT = E_btoE_d(Path2_number, DFT_dataset_sub)
            Path2_barrier_TL = E_btoE_d(Path2_number, TL_dataset_sub)
            Path3_barrier_DFT = E_btoE_d(Path3_number, DFT_dataset_sub)
            Path3_barrier_TL = E_btoE_d(Path3_number, TL_dataset_sub)
            # Find the energy curves of the three paths
            Path1_profile_DFT = E_dtoProfile(Path1_barrier_DFT)
            Path1_profile_TL = E_dtoProfile(Path1_barrier_TL)
            Path2_profile_DFT = E_dtoProfile(Path2_barrier_DFT)
            Path2_profile_TL = E_dtoProfile(Path2_barrier_TL)
            Path3_profile_DFT = E_dtoProfile(Path3_barrier_DFT)
            Path3_profile_TL = E_dtoProfile(Path3_barrier_TL)
            # Find the path peak
            Path1_profile_max_DFT = profile_peak(Path1_profile_DFT)
            Path1_profile_max_TL = profile_peak(Path1_profile_TL)
            Path2_profile_max_DFT = profile_peak(Path2_profile_DFT)
            Path2_profile_max_TL = profile_peak(Path2_profile_TL)
            Path3_profile_max_DFT = profile_peak(Path3_profile_DFT)
            Path3_profile_max_TL = profile_peak(Path3_profile_TL)

            Path1_name_DFT = doped_surface + '_Path1_DFT'
            Path2_name_DFT = doped_surface + '_Path2_DFT'
            Path3_name_DFT = doped_surface + '_Path3_DFT'
            Path1_name_TL = doped_surface + '_Path1_TL'
            Path2_name_TL = doped_surface + '_Path2_TL'
            Path3_name_TL = doped_surface + '_Path3_TL'

            Path1_profile_outfile_DFT = os.path.join(IVB_VB_binding_energy_path, Path1_name_DFT + '.txt')
            Path2_profile_outfile_DFT = os.path.join(IVB_VB_binding_energy_path, Path2_name_DFT + '.txt')
            Path3_profile_outfile_DFT = os.path.join(IVB_VB_binding_energy_path, Path3_name_DFT + '.txt')
            Path1_profile_outfile_TL = os.path.join(IVB_VB_binding_energy_path, Path1_name_TL + '.txt')
            Path2_profile_outfile_TL = os.path.join(IVB_VB_binding_energy_path, Path2_name_TL + '.txt')
            Path3_profile_outfile_TL = os.path.join(IVB_VB_binding_energy_path, Path3_name_TL + '.txt')
            Total_barrier_DFT = os.path.join(IVB_VB_binding_energy_path, 'total_barrier_DFT.txt')
            Total_barrier_TL = os.path.join(IVB_VB_binding_energy_path, 'total_barrier_TL.txt')

            with open(Total_profile_peak_DFT, 'a') as f:
                f.write(str(Path1_profile_max_DFT))
                f.write("\n")
                f.write(str(Path2_profile_max_DFT))
                f.write("\n")
                f.write(str(Path3_profile_max_DFT))
                f.write("\n")

            with open(Total_profile_peak_TL, 'a') as f:
                f.write(str(Path1_profile_max_TL))
                f.write("\n")
                f.write(str(Path2_profile_max_TL))
                f.write("\n")
                f.write(str(Path3_profile_max_TL))
                f.write("\n")

"""







