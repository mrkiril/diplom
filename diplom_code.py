import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy.interpolate import UnivariateSpline
from scipy.interpolate import griddata
#import pylab
import scipy
import scipy.fftpack
from scipy.ndimage.filters import gaussian_filter
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
import copy
import json

kb = 8.617 * 10**(-5)  # стала больцмана
data_dic = {
    "file_name": "Example name",
    "x_name": "degree",
    "x_size": 480,
    "x_scale": [],
    "y_name": "eV",
    "y_size": 640,
    "y_scale": [],
    "data": [],
    #"data_curvave": [],
}
#      file           edc_line maximim_numbers curv_param
file_arr = [
    #("data_igor/BKFA_049.txt",),  # BKFA
    #("data_igor/BKFA_005.txt",),  # BSCCO
    #("data_igor/BKFA_035.txt",),  # BSCCO
    #("data_igor/FeSeTe.txt",),    # FeSeTe
    ("data_igor/FSS3_001.txt", 230, 2, (0.1, 0.01, 0.1), (0, 198), "LH"),      # 62.93K
    ("data_igor/FSS3_002.txt", 230, 2, (0.1, 0.01, 0.1), (0, 198), "LV"),      # 38.51K      27K
    ("data_igor/FSS3_003.txt", 230, 2, (0.1, 0.01, 0.1), (0, 198), "LH"),      # 30.37K
    #("data_igor/FSS3_004.txt",), # графік лайно
    #("data_igor/FSS3_005.txt",), # графік лайно
    ("data_igor/FSS3_006.txt", 230, 2, (0.1, 0.01, 0.1), (0, 158), "LH"),      # 40.2K
    ("data_igor/FSS3_007.txt", 230, 2, (0.1, 0.01, 0.1), (0, 158), "LH"),      # 43.65K      35
    ("data_igor/FSS3_008.txt", 230, 2, (0.1, 0.01, 0.1), (0, 158), "LH"),      # 49.15K      46
    ("data_igor/FSS3_009.txt", 230, 2, (0.1, 0.01, 0.1), (0, 158), "LV"),      # 57.3K
    ("data_igor/FSS3_010.txt", 230, 2, (0.1, 0.01, 0.1), (0, 159), "LV"),      # 61.3K       51
    ("data_igor/FSS3_011.txt", 230, 2, (0.1, 0.01, 0.1), (0, 158), "LH"),      # 52.6K
    ("data_igor/FSS3_012.txt", 230, 2, (0.1, 0.01, 0.1), (0, 157), "LH"),      # 53.9K       56
    ("data_igor/FSS3_013.txt", 230, 2, (0.1, 0.01, 0.1), (0, 154), "LV"),      # 65.61K
    ("data_igor/FSS3_014.txt", 230, 2, (0.1, 0.01, 0.1), (0, 152), "LV"),      # 72.96K      63
    ("data_igor/FSS3_015.txt", 230, 2, (0.1, 0.01, 0.1), (0, 155), "LH"),      # 58.42K      62
    ("data_igor/FSS3_016.txt", 230, 2, (2.0, 0.01, 0.2), (0, 157), "LH"),      # 60.92K
    ("data_igor/FSS3_017.txt", 230, 2, (1.0, 0.01, 0.1), (0, 156), "LH"),      # 62.15K      66
    ("data_igor/FSS3_018.txt", 230, 2, (0.1, 0.01, 0.1), (0, 158), "LV"),      # 179.8K
    ("data_igor/FSS3_019.txt", 230, 2, (0.01, 0.01, 0.01), (0, 158), "LV"),    # 178.1K      72
    ("data_igor/FSS3_020.txt", 230, 2, (0.01, 0.01, 0.1),  (0, 159), "LV"),    # 177.9K
    ("data_igor/FSS3_021.txt", 230, 2, (0.3, 0.01, 0.1),   (0, 156), "LH"),    # 67.79K
    ("data_igor/FSS3_022.txt", 230, 2, (0.1, 0.01, 0.1),   (0, 156), "LH"),    # 67.64K
    ("data_igor/FSS3_023.txt", 230, 2, (0.05, 0.01, 0.1),  (0, 158), "LH"),    # 67.48K
    ("data_igor/FSS3_024.txt", 230, 2, (0.01, 0.01, 0.01), (0, 155), "LV"),    # 175.6K
    ("data_igor/FSS3_025.txt", 230, 2, (0.1, 0.01, 0.1),   (0, 155), "LV"),    # 173.6K      77
    ("data_igor/FSS3_026.txt", 230, 2, (0.01, 0.01, 0.1),  (0, 160), "LH"),    # 75.25K
    
    ("data_igor/FSS3_027.txt", 230, 1, (8.0, 0.01, 8.0),   (0, 160), "LH"),    # 158.5K
    ("data_igor/FSS3_028.txt", 230, 1, (10.0, 0.01, 10.0), (0, 160), "LV"),    # 178K
    ("data_igor/FSS3_029.txt", 230, 1, (10.0, 0.01, 10.0), (0, 160), "LV"),    # 178.17K
    ("data_igor/FSS3_030.txt", 280, 1, (10.1, 0.01, 10.1), (0, 160), "LH"),    # 87K
    ("data_igor/FSS3_031.txt", 230, 2, (0.1, 0.01, 0.1),   (0, 160), "LH"),    # 61.67K
    ("data_igor/FSS3_032.txt", 230, 2, (0.1, 0.01, 0.1),   (0, 160), "LV"),    # 106.7
    ("data_igor/FSS3_033.txt", 230, 2, (0.1, 0.01, 0.1),   (0, 160), "LV"),    # 103
    ("data_igor/FSS3_034.txt", 230, 2, (0.1, 0.01, 0.1),   (0, 160), "LH"),    # 60.79
    ("data_igor/FSS3_035.txt", 230, 2, (0.1, 0.01, 0.1),   (0, 160), "LV"),    # 91.92
    ("data_igor/FSS3_036.txt", 230, 2, (2.2, 0.01, 0.6),   (0, 160), "LH"),    # 57.47
    ("data_igor/FSS3_037.txt", 230, 2, (0.1, 0.01, 0.1),   (0, 160), "LV"),    # 95.8
    ("data_igor/FSS3_038.txt", 230, 2, (0.1, 0.01, 0.1),   (0, 160), "LV"),    # 59.3
    ("data_igor/FSS3_039.txt", 120, 2, (0.01, 0.01, 0.01), (0, 160), "LV"),    # 52.32
    ("data_igor/FSS3_040.txt", 220, 2, (0.01, 0.01, 0.01), (0, 160), "LH"),    # 48.70
    #("data_igor/FSS3_041.txt"), # графік лайно 
    ("data_igor/FSS3_042.txt", 230, 2, (0.1, 0.01, 0.1),   (0, 160), "LH")     # 54.33
]

def scale_to_list(strng):
    return [float(el) for el in strng.split(" ")]

def get_data_from_file(set_str):
    config = configparser.ConfigParser()
    config.read_string(set_str)
    if "Region 1" in config:
        conf = config['Region 1']
        if 'Dimension 1 name' in conf:
            data_dic["y_name"] = conf["Dimension 1 name"]
        if 'Dimension 1 size' in conf:
            data_dic["y_size"] = int(conf["Dimension 1 size"])
        if 'Dimension 1 scale' in conf:
            data_dic["y_scale"] = scale_to_list(conf["Dimension 1 scale"])
        if 'Dimension 2 name' in conf:
            data_dic["x_name"] = conf["Dimension 2 name"]
        if 'Dimension 2 size' in conf:
            data_dic["x_size"] = int(conf["Dimension 2 size"])
        if 'Dimension 2 scale' in conf:
            data_dic["x_scale"] = scale_to_list(conf["Dimension 2 scale"])
    return "azaza"

def point_arr_creator():
    x_arr = []
    y_arr = []
    z_arr = []
    print("y_scale >> ", len(data_dic["y_scale"]))
    print("x_scale >> ", len(data_dic["x_scale"]))

    print()
    # print(data_dic["data"])
    print("DATA y_scale >> ", len(data_dic["data"]))
    print("DATA x_scale >> ", len(data_dic["data"][0]))

    for i in range(len(data_dic["y_scale"])):
        y = data_dic["y_scale"][i]
        for j in range(len(data_dic["x_scale"])):
            x = data_dic["x_scale"]
            x_arr.append(x)
            y_arr.append(y)
            z_arr.append(data_dic["data"][i][j])

    return (x_arr, y_arr, z_arr)

def get_data_from_file_igor(path):
    data = None
    data_dic["data"] = []
    data_dic["x_scale"] = []
    data_dic["y_scale"] = []
    with open(path, 'r') as fp:
        lines = fp.readlines()
        for i in range(len(lines)):
            line = lines[i]
            if(i == 0):
                data_dic["file_name"] = line.replace("\t", "")

            if(i == 2):
                #print( line.encode() )
                arr = line[2:].strip().split("\t")
                data_dic["x_scale"] = [float(e) for e in arr]
                data_dic["x_size"] = len(arr)

            # data
            if(i > 2):
                arr = line[1:].strip().split("\t")
                arr = [float(e) for e in arr]
                data_dic["y_scale"].append(arr[0])
                data_dic["data"].append(arr[1:])

    data_dic["y_size"] = len(data_dic["data"])
    data_dic["data"].reverse()
    data_dic["y_scale"].reverse()

def plot3D(X, Y, Z):
    # Color maps
    # cm.terrain
    # cm.jet
    # cm.coolwarm
    # cm.Spectral
    x_matrix = X
    y_matrix = Y
    z_matrix = np.array(Z)

    print("x_matrix X >> ", len(x_matrix), " Y >> ", len(x_matrix[0]))
    print("y_matrix X >> ", len(y_matrix), " Y >> ", len(y_matrix[0]))
    print(len(z_matrix), len(z_matrix[0]))

    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_surface(x_matrix, y_matrix, z_matrix, cmap=cm.terrain)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    #cset = ax.contourf(x_matrix, y_matrix, z_matrix, zdir='z', offset=-140, cmap=cm.coolwarm)
    #cset = ax.contourf(x_matrix, y_matrix, z_matrix, zdir='x', offset=25, cmap=cm.coolwarm)
    #cset = ax.contourf(x_matrix, y_matrix, z_matrix, zdir='y', offset=-25, cmap=cm.coolwarm)

    # легенда кольору який колір чому відповідає
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.title(data_dic["file_name"])
    plt.show()

def plot2D(data=data_dic["data"], ax_x=data_dic["x_scale"], ax_y=data_dic["y_scale"], cmap="terrain", title="Заголовок"):
    # Color maps
    # cm.terrain
    # cm.jet
    # cm.coolwarm
    # cm.Spectral
    # cm.greys
    def x_axix_foo():
        print(len(data[0]))
        print(len(ax_x) )
        xx = np.arange(0, len(data[0]), X_STEP)
        xlabels = [str(round(ax_x[i], 3)) for i in xx]
        return (xx, xlabels)

    def y_axix_foo():
        yy = np.arange(0, len(data), Y_STEP)
        ylabels = [str(round(ax_y[i], 3)) for i in yy]
        return (yy, ylabels)

    X_STEP = 80
    Y_STEP = 80
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_title(title, color='black', fontsize=16)
    ax.set_ylabel('E, eV', color='black', fontsize=16)
    ax.set_xlabel('k, 1/A', color='black', fontsize=16)

    # Colorbar
    #
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im = ax.imshow(data, cmap=cmap)
    fig.colorbar(im, cax=cax, orientation='vertical')

    xx, xlabels = x_axix_foo()
    ax.set_xticks(xx)
    ax.set_xticklabels(xlabels, color='black', rotation=0, fontsize=10)

    yy, ylabels = y_axix_foo()
    ax.set_yticks(yy)
    ax.set_yticklabels(ylabels, color='black', rotation=0, fontsize=10)

    plt.show()

def plot1D(X, *args):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for ar in args:
        line = ax.plot(X, ar, color='black', linewidth=1)
    plt.show()

def grid_data(X, Y, Z):
    # data is 2D array
    #
    def func(x, y):
        return x * (1 - x) * np.cos(4 * np.pi * x) * np.sin(4 * np.pi * y**2)**2

    x_matrix, y_matrix = np.meshgrid(X, Y)
    z_matrix = np.array(Z)

    print("len X >> ", len(X))
    print("len Y >> ", len(Y))

    points = []
    values = []
    for j in range(len(Y)):
        for i in range(len(X)):
            ar = [X[i], Y[j]]
            points.append(ar)
            values.append(Z[j][i])

    points = np.array(points)
    values = np.array(values)

    #grid_x, grid_y = np.mgrid[0:1:100j, 0:1:100j]
    #points = np.random.rand(1000, 2)
    #values = func(points[:,0], points[:,1])
    #print("grid_x >> ", len(grid_x), "grid_x[0] len >> ", len(grid_x[0]))
    # print()
    #print("points len >> ", len(points))

    grid_x = x_matrix
    grid_y = y_matrix

    grid_z0 = griddata(points, values, (grid_x, grid_y), method='nearest')
    grid_z1 = griddata(points, values, (grid_x, grid_y), method='linear')
    grid_z2 = griddata(points, values, (grid_x, grid_y), method='cubic')

    plt.subplot(221)
    plt.imshow(z_matrix, origin='lower')
    plt.plot(points[:, 0], points[:, 1], 'k.', ms=1)
    plt.title('Original')

    plt.subplot(222)
    plt.imshow(grid_z0,  origin='lower')
    plt.title('Nearest')

    plt.subplot(223)
    plt.imshow(grid_z1,  origin='lower')
    plt.title('Linear')

    plt.subplot(224)
    plt.imshow(grid_z2,  origin='lower')
    plt.title('Cubic')
    plt.gcf().set_size_inches(10, 10)
    plt.show()

def smooth_1D(y, box_pts):
    # mode same
    # full, same, valid
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def DOS(shear):
    """
    1. Calculate DOS
    2. Normalize the data in >> data_dic["data"]
    3. Recalculate DOS
    4. Fiting and take Fermi lvl
    """
    def foo(x, A, B, C, D, E):
        """
        Функція взята з файла Гайд По Ігорю
        який мені дав топоров
        """
        # return  (A+E*(x-B))/(1+np.exp((x-B)/C))+D
        return (A + E * (x - B)) / (1 + np.exp((x - B) / (kb * C))) + D
        #return (A) / (1 + np.exp((x - B) / (kb * C))) + D

    def ini_params_foo(dos):
        def fermi_foo():
            """
            calculate nesrest fermi level before fiting
            """
            fermi = data_dic["y_scale"][gradient_dos_max_ind]
            return fermi

        # INI_PARAM = [4, 15.64, 0.0001, 10, 0] #початкові значення для DOS
        INI_PARAM[0] = max(dos[sh_l:sh_r])      # A
        INI_PARAM[1] = fermi_foo()   # B
        # INI_PARAM[2]               # C
        # INI_PARAM[3] = max(dos)    # D
        # INI_PARAM[4] =             # E
        pass

    def dos_foo():
        return [sum(y_arr) for y_arr in data_dic["data"]]

    def normalize(dos):
        """
        dos_rp - dos right part in dos()dx 
        """
        dos_rp = dos[:gradient_dos_max_ind]
        condition = np.amin(dos_rp) + \
            ((np.amax(dos_rp) - np.amin(dos_rp)) / 10)

        NoK = np.zeros(data_dic["x_size"])
        for i in range(len(dos_rp)):
            if dos_rp[i] < condition:
                NoK += data_dic["data"][i]

        #plot1D(data_dic["x_scale"], NoK)
        # Cheak element for not 0 numbers
        #
        for i in range(len(NoK)):
            if NoK[i] == 0 or np.isnan(NoK[i]):
                if i == 0:
                    NoK[0] = NoK[1]
                if i == len(NoK) - 1:
                    NoK[-1] = NoK[-2]

        tmp_data = np.array(copy.deepcopy(data_dic["data"]))
        arr = []
        for el in tmp_data:
            arr.append(el / NoK)

        return np.array(arr)

    def shear_foo(dos):
        """
        Повертає значення зрізу в якому шукатиметься DOS
        """
        def list_max_foo():
            """
            Побудова відсортованого массива за максимальними значеннями піків
            """
            list_max = argrelextrema(gradient_dos, np.greater_equal, order=4, mode="clip")[0]
            dtype = [('ind', int), ('val', float)]
            values = [(l, gradient_dos[l]) for l in list_max]
            a = np.array(values, dtype=dtype)       # create a structured array
            a = [l for l in a if l[1] > NOISE_LVL and l[0] > 50]
            #
            # Лишити леше 1 максимальний пік, самий правий
            ls_max = [l for l in a if l[1] < 0.98]
            max_09 = [l for l in a if l[1] > 0.98]
            max_09 = np.sort(max_09[:], order='ind')
            #print(max_09[-1])
            ls_max.append(max_09[-1])
            #ls_max = np.array(values, dtype=dtype)
            ls_max = np.sort(ls_max[:], order='ind')
            return ls_max
        
        def list_min_foo():
            """
            Побудова відсортованого массива за мінімального значеннями піків
            """
            dtype = [('ind', int), ('val', float)]
            list_min = argrelextrema(gradient_dos, np.less, order=4, mode="clip")[0]
            values = [(l, gradient_dos[l]) for l in list_min]
            a = np.array(values, dtype=dtype)       # create a structured array
            ls_min = np.sort(a, order='ind')
            return ls_min

        NOISE_LVL = 0.2  # не пропускати максимумми нижче цього рівня
        dos = smooth_1D(dos, 3)
        dos[-4] = dos[-5]
        dos[-3] = dos[-4]
        dos[-2] = dos[-3]
        dos[-1] = dos[-2]
        gradient_dos = np.gradient(dos)
        gradient_dos = smooth_1D(gradient_dos, 8)
        gradient_dos = gradient_dos / np.amax(gradient_dos)

        
        ls_max = list_max_foo()
        l_max_typle = ls_max[0]
        
        ls_min = list_min_foo()
        #print(ls_min)
        # Пошук найближчого правого мінімума
        # від першого максимума
        # для виділення області фітування для DOS
        # так як мінімум похідної означетиме закінчення області максимума на
        # DOS
        l_min_typle = 0
        for l in ls_min:
            if l_max_typle[0] - l[0] < 0:
                l_min_typle = l
                break
        #print(l_min_typle)
        # Візуалізація
        # DOS і похідна на одному графіку
        # Крапочки відмічають місця до яких фітуватиметься
        
        X = [i for i in range(len(dos))]
        fig = plt.figure()
        ax = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        ax.set_title(FILE_PATH, color='black', fontsize=12)
        ax.plot(X, dos, color='black', linewidth=1)
        #ax.scatter(final_ind, dos[final_ind], color="red", s=15)
        #ax.plot([ for  ] )

        ax2.plot(X, gradient_dos, color='green', linewidth=1)
        ax2.plot(X, np.zeros(len(dos)), color='black', linewidth=0.5)
        for l in ls_max:
            ax2.scatter(l[0], l[1], color="red", s=10)

        
        ax2.scatter(l_min_typle[0], l_min_typle[1], color="blue", s=10)                
        plt.show()
        
        return (0, l_min_typle[0])
        

    dos = dos_foo()
    gradient_dos = np.gradient(dos)
    gradient_dos_max_ind = np.argmax(gradient_dos)
    INI_PARAM = [4, 22.674, 40, 1, 0]  # початкові значення для DOS

    tmp = normalize(dos)
    #normilise_graph(original=data_dic["data"], normalize=tmp)    
    data_dic["data"] = tmp
    dos = dos_foo()
    #sh_l, sh_r = shear_foo(dos)
    sh_l, sh_r = shear
    print("SHEAR  ({}, {})".format(sh_l, sh_r))

    ini_params_foo(dos)

    """
    print("max(dos) >> ", max(dos))
    print("max(dos) - >> ", max(dos)-max(dos)*15/100)
    print("max(dos) + >> ", max(dos)+max(dos)*15/100)
    print("maybe >> B - >> ", INI_PARAM[1]-INI_PARAM[1]*0.02/100)
    print("maybe >> B + >> ", INI_PARAM[1]+INI_PARAM[1]*0.02/100)
    print()
    """
    A_min = INI_PARAM[0] - INI_PARAM[0] * 0.9
    A_max = INI_PARAM[0] + INI_PARAM[0] * 0.9
    B_min = INI_PARAM[1] - INI_PARAM[1] * 2 * 10**(-3)
    B_max = INI_PARAM[1] + INI_PARAM[1] * 6 * 10**(-3)
    C_min = 0.0
    C_max = 273
    D_min = 0.0
    D_max = A_max
    E_min = -np.inf
    E_max = np.inf

    #             A      B      C      D      E
    bounds_min = [A_min, B_min, C_min, D_min, E_min]
    bounds_max = [A_max, B_max, C_max, D_max, E_max]
    #popt, pcov = curve_fit(foo, data_dic["y_scale"], dos, p0=INI_PARAM)
    popt, pcov = curve_fit(foo, data_dic["y_scale"][sh_l:sh_r], dos[
                           sh_l:sh_r], bounds=(bounds_min, bounds_max))
    # print("DOS")
    print("A= {:<10}  {:<7} <A< {:<5}".format(
        round(popt[0], 4), round(A_min, 3), round(A_max, 3)))
    print("B= {:<10}  {:<7} <B< {:<5}".format(
        round(popt[1], 4), round(B_min, 3), round(B_max, 3)), "fermi lvl")
    print("C= {:<10}  {:<7} <C< {:<5}".format(
        round(popt[2], 4), round(C_min, 3), round(C_max, 3)), "temperature")
    print("D= {:<10}  {:<7} <D< {:<5}".format(
        round(popt[3], 4), round(D_min, 3), round(D_max, 3)))
    print("E= {:<10}  {:<7} <E< {:<5}".format(
        round(popt[4], 4), round(E_min, 3), round(E_max, 3)))
    print(" ")

    #plot1D(data_dic["y_scale"][sh_l:sh_r], dos[sh_l:sh_r])
    #plot1D(data_dic["y_scale"][sh_l:sh_r], dos[sh_l:sh_r], foo(data_dic["y_scale"][sh_l:sh_r], *popt))
    # Візуалізація DOS з фітом і рисочкою рівня фермі на ньому
    """
    X = data_dic["y_scale"]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("FeSe, обробка DOS фермі сходинкою", color='black', fontsize=22)
    ax.set_ylabel('Intensity', color='black', fontsize=22)
    ax.set_xlabel('E, eV', color='black', fontsize=22)

    #ax.set_title(FILE_PATH, color='black', fontsize=12)
    ax.plot(X, dos, color='grey', linewidth=5, label="DOS")
    #ax.plot(X[sh_l:sh_r], dos[sh_l:sh_r], color='black', linewidth=1, label="DOS")
    ax.plot(X[sh_l:sh_r], foo(data_dic["y_scale"][sh_l:sh_r], *popt), color='black', linewidth=5.5, label="Апроксимація DOS")
    ax.plot([popt[1], popt[1]], [0, max(dos[sh_l:sh_r])], color='blue', linewidth=3.5)
    
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(20) 

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(20) 
    ax.text(popt[1], max(dos[sh_l:sh_r]), 'F='+str(round(popt[1], 4))+ "  T="+str(round(popt[2], 4)), color='black', fontsize=22)
    plt.legend(fontsize=22)
    #plt.grid(True)
    plt.show()
    """

    return (popt[1], popt[2])

def axis_recalculate(ef):
    """
    Recalculate X and Y axis
    ef - efrmi energi
    """
    h = 1.054 * 10**(-34)
    Ek = ef * 1.60217733 * 10**(-19)
    me = 9.1093826 * 10**(-31)
    c = 3 * 10**(8)

    def foo_x(x):
        return ((np.pi / (180 * h * 10**10)) * (np.sqrt(Ek * 2 * me))) * x

    def foo_y(y):
        return y - ef
    #print( foo_x(1) )
    #print( data_dic["x_scale"][:5] )
    #print( [foo_x(el) for el in data_dic["x_scale"][:5] ]  )
    # k0=np.pi/180*np.sqrt(2*me*Ek)/6.62607e-34/10**10*2*np.pi
    #print( k0 )
    #
    #
    ax_x = [foo_x(el) for el in data_dic["x_scale"][:]]
    ax_y = [foo_y(el) for el in data_dic["y_scale"][:]]
    return (ax_x, ax_y)

def fiting_MDC():
    def foo_lor_2(x, A, B, C, D, E, F, G, H):
        """
        Функція взята з файла Гайд По Ігорю
        який мені дав топоров - Лоренціана
        C*E/((x-D)**2+E )

        С- амплітуда
        Е - квадрат напівширини
        D - положення максимума по осі Х
        """
        # return  (A+E*(x-B))/(1+np.exp((x-B)/C))+D
        return (A + (B * (x**2)) + (C * E / (E + (x - D)**2)) + (F * H / ((x - G)**2) + H))

    def ini_params_foo(y_data):
        y_data = smooth_1D(y_data, 85)
        list_max = argrelextrema(y_data, np.greater, order=50)[0]
        if len(list_max) == 2:
            INI_PARAM = [0, 0, 0, 0, 0, 0, 0, 0]  # початкові значення для DOS

        if len(list_max) == 4:
            # початкові значення для DOS
            INI_PARAM = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        # A  B  C  D  E  F  G  H
        # 0  1  2  3  4  5  6  7
        ind_1 = 2
        ind_2 = 3
        ind_3 = 4
        for x_pixel in argrelextrema(y_data, np.greater, order=10)[0]:
            x = data_dic["x_scale"][x_pixel]
            y = y_data[x_pixel]
            print(x, "x", y)

            INI_PARAM[ind_1] = y       # C - Амплітуда
            INI_PARAM[ind_2] = x       # D - Положення максимума по ос Х
            INI_PARAM[ind_3] = 0       # E - Півширина HWHM

            ind_1 = ind_1 + 3
            ind_2 = ind_2 + 3
            ind_3 = ind_3 + 3
        # INI_PARAM[0] = max(dos)     # A
        # INI_PARAM[1] = fermi_foo()  # B
        # INI_PARAM[2]               # C
        # INI_PARAM[3] = max(dos)    # D
        # INI_PARAM[4] =             # E
        return INI_PARAM

    def bounds_foo(INI_PARAM):
        A_min = -np.inf
        A_max = np.inf
        B_min = -np.inf
        B_max = np.inf
        C_min = INI_PARAM[2] - INI_PARAM[2] * 0.1
        C_max = INI_PARAM[2] + INI_PARAM[2] * 0.1
        D_min = INI_PARAM[3] - INI_PARAM[3] * 0.1
        D_max = INI_PARAM[3] + INI_PARAM[3] * 0.1
        E_min = 0
        E_max = np.inf
        F_min = INI_PARAM[5] - INI_PARAM[5] * 0.1
        F_max = INI_PARAM[5] + INI_PARAM[5] * 0.1
        G_min = INI_PARAM[6] - INI_PARAM[6] * 0.1
        G_max = INI_PARAM[6] + INI_PARAM[6] * 0.1
        H_min = 0
        H_max = np.inf

        bounds_min = [A_min, B_min, C_min, D_min, E_min, F_min, G_min, H_min]
        bounds_max = [A_max, B_max, C_max, D_max, E_max, F_max, G_max, H_max]
        print(C_min, " < C < ", C_max)
        print(D_min, " < D < ", D_max)
        print(E_min, " < E < ", E_max)
        print(F_min, " < F < ", F_max)
        print(G_min, " < G < ", G_max)
        print(H_min, " < H < ", H_max)
        return (bounds_min, bounds_max)

    x_axis = np.array(data_dic["x_scale"][:430])
    y_data = data_dic["data"][60][:430]
    y_data_smooth = smooth_1D(y_data, 20)

    INI_PARAM = ini_params_foo(y_data)
    bounds_min, bounds_max = bounds_foo(INI_PARAM)

    #popt, pcov = curve_fit(foo_lor_2, x_axis, y_data, bounds=(bounds_min, bounds_max))
    popt, pcov = curve_fit(foo_lor_2, x_axis, y_data_smooth)
    print("A >> ", popt[0])
    print("B >> ", popt[1])
    print("C >> ", popt[2])
    print("D >> ", popt[3])
    print("E >> ", popt[4])
    print("F >> ", popt[4])
    print("G >> ", popt[4])
    print("H >> ", popt[4])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x_axis, y_data_smooth, color='black', linewidth=1)
    ax.plot(x_axis, foo_lor_2(x_axis, *popt), color='red', linewidth=1)

    for x_pixel in argrelextrema(y_data_smooth, np.greater, order=10)[0]:
        x = data_dic["x_scale"][x_pixel]
        y = y_data[x_pixel]
        print(x, "x", y)
        ax.scatter(x, y, color="red", s=15)

    plt.show()
    # return popt[1]

def curvature(coef_cx, coef_cy, coef_i):
    def to_zero(cv):
        cv_copy = copy.deepcopy(cv)
        #maxx = np.amax(cv_copy)/MAX_LIMIT
        for i in range(len(cv_copy)):
            for j in range(len(cv_copy[i])):
                cv_copy[i][j] *= -1

        for i in range(len(cv_copy)):
            for j in range(len(cv_copy[i])):
                if cv_copy[i][j] < 0:
                    cv_copy[i][j] = 0
                # if cv_copy[i][j] > 0 and cv_copy[i][j] < maxx:
                #    cv_copy[i][j] = 0
        return cv_copy

    """
    T_temp=((factor*avg+weight*t_diff1math*t_diff1math)*t_diff2matv-2*weight*t_diff1math*t_diff1matv*t_diff2matvh+weight*(factor*avg+t_diff1matv*t_diff1matv)*t_diff2math)/(factor*avg+weight*t_diff1math*t_diff1math+t_diff1matv*t_diff1matv)^1.5

    N1 = factor*avg+weight*t_diff1math*t_diff1math)*t_diff2matv
    N2 = -2*weight*t_diff1math*t_diff1matv*t_diff2matvh
    N3 = weight*(factor*avg+t_diff1matv*t_diff1matv)*t_diff2math)
    N4 = (factor*avg+weight*t_diff1math*t_diff1math+t_diff1matv*t_diff1matv)^1.5


    """
    data = np.array(data_dic["data"][:])
    data = gaussian_filter(data, sigma=GAUS_COEF)

    #data = np.array([[2, 5, 7], [3, 6, 8], [4, 7, 9] ])
    #print( data*np.array([[2, 2, 2], [2, 2, 2], [2, 2, 2] ]) )
    dx = abs((data_dic["x_scale"][1] - data_dic["x_scale"][0]))
    dy = abs((data_dic["y_scale"][1] - data_dic["y_scale"][0]))
    weight = (dx / dy) * (dx / dy)

    df_dx, df_dy = np.gradient(data, dy, dx)
    d2f_dx2, d2f_dxdy = np.gradient(df_dx, dy, dx)
    _, d2f_dy2 = np.gradient(df_dy, dy, dx)
    avgv = abs(np.amin(df_dy))
    avgh = abs(np.amin(df_dx))
    #print("df_dx >> ", np.amin(df_dx))
    #print("df_dy >> ", np.amin(df_dy))
    #print("d2f_dy2 >> ", np.amin(d2f_dy2))
    #print("d2f_dx2 >> ", np.amin(d2f_dx2))

    psi = (data_dic["x_scale"][1] - data_dic["x_scale"][0])**2
    niy = (data_dic["y_scale"][1] - data_dic["y_scale"][0])**2
    I0 = coef_i * np.amax(data)**2
    Cx = I0 * psi * coef_cx
    Cy = I0 * niy * coef_cy
    #print("I0 >> ", I0, " psi >> ", psi, " niy >> ", niy)
    #print("Cx >> ", Cx)
    #print("Cy >> ", Cy)
    #print("tmp >> ", np.amin(  (d2f_dy2)    ))

    N1 = (1 + Cx * df_dx**2) * Cy * d2f_dy2
    N2 = 2 * Cx * Cy * df_dx * df_dy * d2f_dxdy
    N3 = (1 + Cy * df_dy**2) * Cx * d2f_dx2
    N4 = (1 + Cx * df_dx**2 + Cy * df_dy**2)**(3 / 2)

    """
    factor = 5
    avg = max(avgv**2, weight*avgh**2)
    C0 = (factor*avg)
    N1 = (C0 + weight*df_dx**2)*d2f_dy2
    N2 = 2 * weight * df_dx * df_dy * d2f_dxdy
    N3 = weight * (C0 + df_dy**2)*d2f_dx2
    N4 = (C0 + weight * df_dx**2 + df_dy**2)**(3/2) 
    """
    #print(" ")
    #print("N1 >> ", np.amin(N1))
    #print("N2 >> ", np.amin(N2))
    #print("N3 >> ", np.amin(N3))
    #print("N4 >> ", np.amin(N4))

    CV = (N1 - N2 + N3) / N4
    CV = to_zero(CV)
    return CV

def curv_plot_2x1():
    """
    Plot raw ARPES data and
    result of curvature working in second windows
    """
    def onChangeValue(value):
        '''!!! Обработчик события изменения значений слайдеров'''
        updateGraph()

    def updateGraph():
        def x_axix_foo():
            xx = np.arange(0, data_dic["x_size"], X_STEP)
            xlabels = [str(round(data_dic["x_scale"][i], 3)) for i in xx]
            return (xx, xlabels)

        def y_axix_foo():
            yy = np.arange(0, data_dic["y_size"], Y_STEP)
            ylabels = [str(round(data_dic["y_scale"][i], 3)) for i in yy]
            return (yy, ylabels)

        X_STEP = 200
        Y_STEP = 200

        print("coef_cx={}, coef_cy={} coef_i={}".format(
            round(coef_cx.val, 3), round(coef_cy.val, 3), round(coef_i.val, 3)))

        original = ax.imshow(data_dic["data"], cmap="Greys")
        CV = curvature(coef_cx.val, coef_cy.val, coef_i.val)
        ax2.clear()
        curv = ax2.imshow(CV, cmap="Greys")

        xx, xlabels = x_axix_foo()
        ax.set_xticks(xx)
        ax.set_xticklabels(xlabels, color='black', rotation=0, fontsize=14)
        ax2.set_xticks(xx)
        ax2.set_xticklabels(xlabels, color='black', rotation=0, fontsize=14)

        yy, ylabels = y_axix_foo()
        ax.set_yticks(yy)
        ax.set_yticklabels(ylabels, color='black', rotation=0, fontsize=14)
        ax2.set_yticks(yy)
        ax2.set_yticklabels(ylabels, color='black', rotation=0, fontsize=14)
        fig.colorbar(original, cax=cax1, orientation='vertical')
        fig.colorbar(curv, cax=cax2, orientation='vertical')
        ax.set_title(" ", color='black', fontsize=16)
        ax.set_ylabel('E, eV', color='black', fontsize=16)
        ax.set_xlabel('k, 1/A', color='black', fontsize=16)

        ax2.set_title(" ", color='black', fontsize=16)
        ax2.set_ylabel('E, eV', color='black', fontsize=16)
        ax2.set_xlabel('k, 1/A', color='black', fontsize=16)

        plt.draw()


    fig = plt.figure()
    fig.subplots_adjust(left=0.07, right=0.95, top=0.95, bottom=0.4)
    ax = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    #ax.set_title(FILE_PATH, color='black', fontsize=12)

    divider1 = make_axes_locatable(ax)
    cax1 = divider1.append_axes('right', size='5%', pad=0.05)

    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes('right', size='5%', pad=0.05)
    
    axes_coef_cx = plt.axes([0.10, 0.25, 0.80, 0.04])
    coef_cx = Slider(axes_coef_cx,
                     label='coef_cx',
                     valmin=0.0,
                     valmax=100.0,
                     valinit=0.001,
                     valfmt='%0.01f')
    coef_cx.on_changed(onChangeValue)

    axes_coef_cy = plt.axes([0.10, 0.17, 0.80, 0.04])
    coef_cy = Slider(axes_coef_cy,
                     label='coef_cy',
                     valmin=0.0,
                     valmax=10.0,
                     valinit=0.001,
                     valfmt='%0.01f',
                     facecolor='c', edgecolor='r')
    coef_cy.on_changed(onChangeValue)

    axes_coef_i = plt.axes([0.10, 0.09, 0.80, 0.04])
    coef_i = Slider(axes_coef_i,
                    label='coef_i',
                    valmin=0.0,
                    valmax=100.0,
                    valinit=0.001,
                    valfmt='%0.01f',
                    facecolor='grey')
    coef_i.on_changed(onChangeValue)
    updateGraph()
    plt.show()

def curv_plot_2x1_dos(foo_edc, edc_line, init):
    """
    Plot raw ARPES data and
    result of curvature working in second windows
    """
    def onChangeValue(value):
        '''!!! Обработчик события изменения значений слайдеров'''
        updateGraph()

    def updateGraph():
        def x_axix_foo():
            xx = np.arange(0, data_dic["x_size"], X_STEP)
            xlabels = [str(round(data_dic["x_scale"][i], 3)) for i in xx]
            return (xx, xlabels)

        def y_axix_foo():
            yy = np.arange(0, data_dic["y_size"], Y_STEP)
            ylabels = [str(round(data_dic["y_scale"][i], 3)) for i in yy]
            return (yy, ylabels)

        X_STEP = 50
        Y_STEP = 30
        #global foo_edc
        print("coef_cx={}, coef_cy={} coef_i={}".format(
            round(coef_cx.val, 3), round(coef_cy.val, 3), round(coef_i.val, 3)))

        CV = curvature(coef_cx.val, coef_cy.val, coef_i.val)
        EDC = [c[230] for c in CV]
        EDC = np.array(EDC) / np.amax(EDC)

        ax.clear()
        ax.set_title(FILE_PATH, color='black', fontsize=12)
        ax.plot(data_dic["y_scale"], EDC, color='black', linewidth=1)
        point = foo_edc(EDC)
        for l in point:
            ax.scatter(l[0], l[1], color="red", s=10)
        ax2.clear()
        curv = ax2.imshow(CV, cmap=cm.terrain)
        ax2.plot([edc_line, edc_line], [0, 640], color='red', linewidth=0.5)

        # Axis settings
        #
        xx, xlabels = x_axix_foo()
        ax2.set_xticks(xx)
        ax2.set_xticklabels(xlabels, color='b', rotation=270, fontsize=6)

        yy, ylabels = y_axix_foo()
        ax2.set_yticks(yy)
        ax2.set_yticklabels(ylabels, color='b', rotation=0, fontsize=6)
        fig.colorbar(curv, cax=cax2, orientation='vertical')
        x = data_dic["x_scale"]
        plt.draw()

    fig = plt.figure()
    fig.subplots_adjust(left=0.07, right=0.95, top=0.95, bottom=0.4)
    ax = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    divider1 = make_axes_locatable(ax)
    cax1 = divider1.append_axes('right', size='5%', pad=0.05)

    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes('right', size='5%', pad=0.05)
    ax.set_title(FILE_PATH, color='black', fontsize=12)
    axes_coef_cx = plt.axes([0.10, 0.25, 0.80, 0.04])
    coef_cx = Slider(axes_coef_cx,
                     label='coef_cx',
                     valmin=0.0,
                     valmax=10.0,
                     valinit=init[0],
                     valfmt='%0.01f')
    coef_cx.on_changed(onChangeValue)

    axes_coef_cy = plt.axes([0.10, 0.17, 0.80, 0.04])
    coef_cy = Slider(axes_coef_cy,
                     label='coef_cy',
                     valmin=0.0,
                     valmax=10.0,
                     valinit=init[1],
                     valfmt='%0.01f',
                     facecolor='c', edgecolor='r')
    coef_cy.on_changed(onChangeValue)

    axes_coef_i = plt.axes([0.10, 0.09, 0.80, 0.04])
    coef_i = Slider(axes_coef_i,
                    label='coef_i',
                    valmin=0.0,
                    valmax=10.0,
                    valinit=init[2],
                    valfmt='%0.01f',
                    facecolor='grey')
    coef_i.on_changed(onChangeValue)

    updateGraph()
    plt.show()

def bias_for_temp():
    """
    перебираю всі спектри FeSeTe
    рахую DOS кожного
    фітую ступенькою фермі
    дістаю температуру
    дістаю координати зон
    засовую їх в ліст
    запаковую це все в джейсон
    зберіаю
    """
    def edc_max(EDC):
        """
        Список максимумів піків на DOS діаграмі
        після обробкою функції curvave
        """
        list_edc_max = argrelextrema(EDC, np.greater, order=10, mode="clip")[0]
        dtype = [('ind', int), ('val', float)]
        values = [(l, EDC[l]) for l in list_edc_max]
        a = np.array(values, dtype=dtype)       # create a structured array
        ls_edc_max = [l for l in a if data_dic["y_scale"][l[0]] > -0.1]
        ls_edc_max = np.sort(ls_edc_max[:], order='val')[-FILE_DATA[2]:]
        ls_edc_max = np.sort(ls_edc_max[:], order='ind')
        # print(ls_edc_max)
        return [(data_dic["y_scale"][l[0]], l[1]) for l in ls_edc_max]

    def json_constr(name, fermi, temp, line_number, line_lvl, polar):
        return {
            "name": name,
            "fermi": fermi,
            "temp": temp,
            "line_num": line_number,
            "line_lvl": line_lvl,
            "polar":polar
        }

    array_fir = []
    array_sec = []
    JSON = {"data": []}
    # arr_line0 = [] #та що нижче
    # arr_line1 = [] #та що вище
    for i in range(0, len(file_arr)):
        print("File path >", file_arr[i][0])
        global FILE_PATH
        global FILE_DATA
        FILE_PATH = file_arr[i][0]
        SHEAR = file_arr[i][4]
        POLAR = file_arr[i][5]
        FILE_DATA = file_arr[i]
        get_data_from_file_igor(FILE_PATH)
        fermi_level, temp = DOS(shear=SHEAR)
        
        CV = curvature(coef_cx=FILE_DATA[3][0], coef_cy=FILE_DATA[3][1], coef_i=FILE_DATA[3][2])
        EDC = [c[FILE_DATA[1]] for c in CV]
        EDC = np.array(EDC) / np.amax(EDC)
        point = edc_max(EDC)
        print(point)

        if len(point) == 1:
            j_son = json_constr(FILE_PATH, fermi_level, temp, 1, point[0][0], POLAR)
            JSON["data"].append(j_son)
        else:
            j_son1 = json_constr(FILE_PATH, fermi_level, temp, 2, point[0][0], POLAR)
            j_son2 = json_constr(FILE_PATH, fermi_level, temp, 1, point[1][0], POLAR)
            JSON["data"].append(j_son1)
            JSON["data"].append(j_son2)

        #curv_plot_2x1_dos(edc_max, FILE_DATA[1], FILE_DATA[3])
    return JSON

def plot_fesete_debug(JSON, line_num=0, t_max=273, t_min=0, polar="LH"):
    """
    FOR DEBUGGING
    {
    'fermi': 18.138, 
    'name': 'data_igor/FSS3_001.txt', 
    'temp': 67.75, 
    'line_lvl': 18.12, 
    'line_num': 1,
    "polar": "LH"
    }
    if line_num == 0 - print all lines
    """ 
    def on_pick(event):
        for i in event.ind:
            data = json_plot[i]
            print(data["name"])
            print("temp=", round(data["temp"], 3), "line_lvl=", round(data["line_lvl"], 4))
            print(" ")
    
    def appnd(l_num):
        json_plot.append(j)
        if j["line_num"] == 1:
            scatter_color.append("black")
        else:
            scatter_color.append("red")

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("FeSeTe, "+"polarization: "+polar, color='black', fontsize=20)
    ax.set_ylabel('E, eV', color='black', fontsize=20)
    ax.set_xlabel('T, K', color='black', fontsize=20)
    scatter_x = []    
    scatter_y = []    
    scatter_color = [] 
    scatter_marker = []
    json_plot = []   
    for j in JSON["data"]:
        if j["temp"] < t_max and j["temp"] > t_min:
            if polar == "all":
                if line_num == 0:
                    appnd(j["line_num"])

                elif line_num == j["line_num"]:
                    appnd(j["line_num"])

                if j["polar"] == "LH":
                    scatter_marker.append("d")
                else:
                    scatter_marker.append("o")

            elif polar == j["polar"]:
                scatter_marker.append("o")
                if line_num == 0:
                    appnd(j["line_num"])

                elif line_num == j["line_num"]:
                    appnd(j["line_num"])
                
            

    x = [a["temp"] for a in json_plot]
    y = [a["line_lvl"] for a in json_plot]
    for i in range(len(x)):
        ax.scatter(x[i], y[i], color=scatter_color[i], picker=5, s=85, marker=scatter_marker[i])
        
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(20) 

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(20)    
    #ax.set_xticklabels(color='black', rotation=0, fontsize=12)
    fig.canvas.mpl_connect('pick_event', on_pick)
    #ax.grid(True)
    #fig.savefig("polar_"+polar+".png", dpi=300)
    plt.show()

def plot_fesete_2D():
    """
    візуалізація зміщення зон на одному графіку
    """
    def y_axis_recalc(fermi_ind):
        STEP = round(data_dic["y_scale"][0] - data_dic["y_scale"][1], 10)
        y_scale = np.zeros(Y_SIZE).reshape((1, -1))
        print("STEP > ", STEP)
        print("fermi_ind > ", fermi_ind)
        
        
        arr1 = np.flipud(np.arange(0, (fermi_ind+1)*STEP, STEP))
        arr2 = np.arange(STEP, (Y_SIZE-fermi_ind)*STEP, STEP)*(-1)
        arr3 = np.append(arr1, arr2)
        print( len(arr1) )
        print( len(arr2) )
        print( len(arr3) )
        print( arr3[98], arr3[99], arr3[100], arr3[101] )
        return arr3

    arr_filter = {
        #("data_igor/FSS3_001.txt",)
        #"data_igor/FSS3_002.txt": {"beet": 240, "beet2": 340, "tup": (5.0, 0.001, 5.0)},
        "data_igor/FSS3_003.txt": {"beet": 240, "beet2": 340, "tup": (2.0, 0.001, 2.0)},    #36K
        "data_igor/FSS3_006.txt":{"beet": 200, "beet2": 340, "tup": (10.0, 0.001, 10.0)},   #27K
        #"data_igor/FSS3_007.txt": {"beet": 240, "beet2": 340, "tup": (5.0, 0.001, 5.0)},
        #"data_igor/FSS3_008.txt",
        #"data_igor/FSS3_009.txt",
        #"data_igor/FSS3_010.txt": {"beet": 200, "tup": (10.5, 0.001, 1.0)},
        #"data_igor/FSS3_011.txt",
        #"data_igor/FSS3_012.txt",
        #"data_igor/FSS3_013.txt": {"beet": 140, "tup": (20.5, 0.001, 1.0)},
        #"data_igor/FSS3_014.txt": {"beet": 140, "tup": (20.5, 0.001, 1.0)},
        #"data_igor/FSS3_015.txt",
        #"data_igor/FSS3_016.txt",
        #"data_igor/FSS3_017.txt",
        #"data_igor/FSS3_018.txt": {"beet": 140, "tup": (20.5, 0.001, 1.0)},
        #"data_igor/FSS3_019.txt": {"beet": 140, "tup": (100.5, 0.001, 1.0)},
        #"data_igor/FSS3_020.txt",
        #"data_igor/FSS3_021.txt": {"beet": 180, "tup": (100.5, 0.001, 1.0)},
        #"data_igor/FSS3_022.txt",
        #"data_igor/FSS3_023.txt",
        #"data_igor/FSS3_024.txt",
        "data_igor/FSS3_025.txt": {"beet": 180, "tup": (2.5, 0.001, 10.0)},              # 77K
        #"data_igor/FSS3_026.txt": {"beet": 180, "tup": (80.5, 0.001, 10.0)},
        #"data_igor/FSS3_027.txt": {"beet": 180, "tup": (100.5, 0.001, 100.0)},
        #"data_igor/FSS3_028.txt": {"beet": 180, "tup": (50.5, 0.001, 50.0)},
        #"data_igor/FSS3_029.txt",
        #"data_igor/FSS3_030.txt",
        #"data_igor/FSS3_031.txt",
        #"data_igor/FSS3_032.txt",
        #"data_igor/FSS3_033.txt",
        #"data_igor/FSS3_034.txt",
        #"data_igor/FSS3_035.txt",
        #"data_igor/FSS3_036.txt",
        #"data_igor/FSS3_037.txt",
        #"data_igor/FSS3_038.txt",
        #"data_igor/FSS3_039.txt",
        #"data_igor/FSS3_040.txt",
        #"data_igor/FSS3_042.txt",
    }

    X_SIZE = data_dic["x_size"]
    Y_SIZE = data_dic["y_size"]
    
    arr_cv_down = []
    arr_cv_up = []
    BASE_index = 100
    BASE_down = np.zeros((Y_SIZE, X_SIZE))
    for i in range(0, len(file_arr)):
        print("File path >", file_arr[i][0])
        if file_arr[i][0] not in arr_filter:
            continue
        global FILE_PATH
        global FILE_DATA
        FILE_PATH = file_arr[i][0]
        FILE_DATA = file_arr[i]
        SHEAR = file_arr[i][4]
        tp = arr_filter[FILE_PATH]["tup"]

        get_data_from_file_igor(FILE_PATH)
        fermi_level, temp = DOS(shear=SHEAR)        
        CV = curvature(coef_cx=tp[0], coef_cy=tp[1], coef_i=tp[2]) 
        
        fermi_ax = np.inf
        fermi_ind = np.inf
        #print( data_dic["y_scale"] )
        for i in range(Y_SIZE):
            el = data_dic["y_scale"][i]
            if abs(el - fermi_level) < fermi_ax:
                fermi_ax = abs(el - fermi_level)
                fermi_ind = i
                #print(fermi_ax, "   ", el - fermi_level, "           ", )
        
        # для нижньої зони
        #
        CV_copy = copy.deepcopy(CV)
        for i in range(len(CV_copy)):
            if i < arr_filter[FILE_PATH]["beet"]:
                CV_copy[i] /= np.inf

        CV_copy = CV_copy / np.amax(CV)
        index = BASE_index
        #print(fermi_ind)
        #print(len(CV_copy))
        for i in range(fermi_ind, fermi_ind+200):
            BASE_down[index] += CV_copy[i]
            index += 1

        print(data_dic["y_scale"][1] - data_dic["y_scale"][0])
        print(data_dic["y_scale"][2] - data_dic["y_scale"][1])
        print(data_dic["y_scale"][3] - data_dic["y_scale"][2])


        # для верхньої зони
        #
        #CV_copy = copy.deepcopy(CV)
        #for i in range(len(CV_copy)):
        #    if i > arr_filter[FILE_PATH]["beet"]:
        #        CV_copy[i] /= np.inf
        #arr_cv_up.append(CV_copy/np.amax(CV))


    ax_x, ax_y = axis_recalculate(fermi_level)
    print("BASE_down > ", len(BASE_down))
    print("BASE_down[0] > ", len(BASE_down[0]))
    print(" ")
    print(" ")
    print("X_SIZE: ", X_SIZE)
    print("Y_SIZE: ", Y_SIZE)
    ax_y = y_axis_recalc(BASE_index)
    #CV_down = np.array(sum(arr_cv_down))
    print(BASE_down.shape)
    plot2D(data=BASE_down, ax_x=ax_x, ax_y=ax_y, cmap="terrain", title=" ") #, cmap="Greys" ax_x=ax_x, ax_y=

    #CV_up = np.array(sum(arr_cv_up))
    #plot2D(CV_up)

def fermi_recalc(JSON):
    for js in JSON["data"]:
        js["line_lvl"] -= js["fermi"]

def chiterstvo(JSON):
    for j in JSON["data"]:
        # LH
        #
        if "data_igor/FSS3_001.txt" == j["name"]:
            j["temp"] = 36
        if "data_igor/FSS3_003.txt" == j["name"]:
            j["temp"] = 36
        if "data_igor/FSS3_004.txt" == j["name"]:
            j["temp"] = 36
        if "data_igor/FSS3_005.txt" == j["name"]:
            j["temp"] = 27
        if "data_igor/FSS3_006.txt" == j["name"]:
            j["temp"] = 27
        if "data_igor/FSS3_007.txt" == j["name"]:
            j["temp"] = 35
        if "data_igor/FSS3_008.txt" == j["name"]:
            j["temp"] = 46
        if "data_igor/FSS3_011.txt" == j["name"]:
            j["temp"] = 51
        if "data_igor/FSS3_012.txt" == j["name"]:
            j["temp"] = 56
        if "data_igor/FSS3_015.txt" == j["name"]:
            j["temp"] = 62
        if "data_igor/FSS3_016.txt" == j["name"]:
            j["temp"] = 62
        if "data_igor/FSS3_017.txt" == j["name"]:
            j["temp"] = 66
        if "data_igor/FSS3_021.txt" == j["name"]:
            j["temp"] = 72
        if "data_igor/FSS3_022.txt" == j["name"]:
            j["temp"] = 72
        if "data_igor/FSS3_023.txt" == j["name"]:
            j["temp"] = 72
        if "data_igor/FSS3_026.txt" == j["name"]:
            j["temp"] = 77
        if "data_igor/FSS3_027.txt" == j["name"]:
            j["temp"] = 77
        if "data_igor/FSS3_030.txt" == j["name"]:
            j["temp"] = 77
        if "data_igor/FSS3_031.txt" == j["name"]:
            j["temp"] = 52
        if "data_igor/FSS3_034.txt" == j["name"]:
            j["temp"] = 47
        if "data_igor/FSS3_036.txt" == j["name"]:
            j["temp"] = 32
        if "data_igor/FSS3_040.txt" == j["name"]:
            j["temp"] = 32
        if "data_igor/FSS3_041.txt" == j["name"]:
            j["temp"] = 32
        if "data_igor/FSS3_042.txt" == j["name"]:
            j["temp"] = 56
        
        # LV
        #
        if "data_igor/FSS3_002.txt" == j["name"]:
            j["temp"] = 36
        if "data_igor/FSS3_009.txt" == j["name"]:
            j["temp"] = 46
        if "data_igor/FSS3_010.txt" == j["name"]:
            j["temp"] = 51
        if "data_igor/FSS3_013.txt" == j["name"]:
            j["temp"] = 56
        if "data_igor/FSS3_014.txt" == j["name"]:
            j["temp"] = 63
        if "data_igor/FSS3_018.txt" == j["name"]:
            j["temp"] = 66
        if "data_igor/FSS3_019.txt" in j["name"]:
            j["temp"] = 72        
        if "data_igor/FSS3_020.txt" in j["name"]:
            j["temp"] = 72 
        if "data_igor/FSS3_024.txt" in j["name"]:
            j["temp"] = 72
        if "data_igor/FSS3_025.txt" in j["name"]:
            j["temp"] = 77
        if "data_igor/FSS3_028.txt" == j["name"]:
            j["temp"] = 77
        if "data_igor/FSS3_029.txt" == j["name"]:
            j["temp"] = 77
        if "data_igor/FSS3_032.txt" == j["name"]:
            j["temp"] = 52
        if "data_igor/FSS3_033.txt" == j["name"]:
            j["temp"] = 47
        if "data_igor/FSS3_035.txt" == j["name"]:
            j["temp"] = 32
        if "data_igor/FSS3_037.txt" == j["name"]:
            j["temp"] = 32
        if "data_igor/FSS3_038.txt" == j["name"]:
            j["temp"] = 32
        if "data_igor/FSS3_039.txt" == j["name"]:
            j["temp"] = 32

def data_filter(JSON):
    """
    Filtering spactrum for final graph
    if element in is arrfilter they will bein final JSON
    """
    arr_filter = [
        #"data_igor/FSS3_001.txt",
        "data_igor/FSS3_002.txt",
        #"data_igor/FSS3_003.txt",
        "data_igor/FSS3_006.txt",
        
        "data_igor/FSS3_007.txt",
        "data_igor/FSS3_008.txt",
        "data_igor/FSS3_009.txt",
        "data_igor/FSS3_010.txt",
        "data_igor/FSS3_011.txt",
        "data_igor/FSS3_012.txt",
        "data_igor/FSS3_013.txt",
        "data_igor/FSS3_014.txt",
        "data_igor/FSS3_015.txt",
        "data_igor/FSS3_016.txt",
        "data_igor/FSS3_017.txt",
        "data_igor/FSS3_018.txt",
        "data_igor/FSS3_019.txt",
        "data_igor/FSS3_020.txt",
        "data_igor/FSS3_021.txt",
        "data_igor/FSS3_022.txt",
        "data_igor/FSS3_023.txt",
        "data_igor/FSS3_024.txt",
        "data_igor/FSS3_025.txt",
        "data_igor/FSS3_026.txt",
        
        #"data_igor/FSS3_027.txt",
        #"data_igor/FSS3_028.txt",
        #"data_igor/FSS3_029.txt",
        #"data_igor/FSS3_030.txt",
        #"data_igor/FSS3_031.txt",
        #"data_igor/FSS3_032.txt",
        #"data_igor/FSS3_033.txt",
        #"data_igor/FSS3_034.txt",
        #"data_igor/FSS3_035.txt",
        #"data_igor/FSS3_036.txt",
        #"data_igor/FSS3_037.txt",
        #"data_igor/FSS3_038.txt",
        #"data_igor/FSS3_039.txt",
        #"data_igor/FSS3_040.txt",
        #"data_igor/FSS3_042.txt",
    ]
    new_json = {"data":[]}
    for j in JSON["data"]:
        if j["name"] in arr_filter:
            new_json["data"].append(j)
    return new_json

# Функції для побудови графіків
# для диплома.
# Викликаються в самому низу
# Вони є самодостатніми. Тобто виклик кожної функції є самодостатнім.
def normilise_graph(original, normalize):
    """
    Два графіка оригінальний і нормований на 1 полотні
    Викликається в функції DOS
    """
    def x_axix_foo(data):
        xx = np.arange(0, len(data[0]), X_STEP)
        xlabels = [str(round(ax_x[i], 3)) for i in xx]
        return (xx, xlabels)

    def y_axix_foo(data):
        yy = np.arange(0, len(data), Y_STEP)
        ylabels = [str(round(ax_y[i], 3)) for i in yy]
        return (yy, ylabels)

    X_STEP = 80
    Y_STEP = 80
    ax_x = data_dic["x_scale"]
    ax_y = data_dic["y_scale"]
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax.set_title("Оригінальний спектр FeSe ", color='black', fontsize=20)
    ax2.set_title("Нормаований FeSe ", color='black', fontsize=20)
    ax.set_ylabel('E, eV', color='black', fontsize=20)
    ax.set_xlabel(r"${\Theta}$, градуси", color='black', fontsize=20)
    ax2.set_xlabel(r"${\Theta}$, градуси", color='black', fontsize=20)

    # Colorbar
    #
    #divider = make_axes_locatable(ax)
    #cax = divider.append_axes('right', size='5%', pad=0.05)
    im = ax.imshow(original, cmap="Greys")
    im = ax2.imshow(normalize, cmap="Greys")
    #fig.colorbar(im, cax=cax, orientation='vertical')

    xx, xlabels = x_axix_foo(original)
    yy, ylabels = y_axix_foo(original)
    ax.set_xticks(xx)
    ax.set_xticklabels(xlabels, color='black', rotation=0, fontsize=16)
    ax.set_yticks(yy)
    ax.set_yticklabels(ylabels, color='black', rotation=0, fontsize=20)    
    ax2.set_xticks(xx)
    ax2.set_xticklabels(xlabels, color='black', rotation=0, fontsize=16)    
    ax2.set_yticks(yy)
    ax2.set_yticklabels(ylabels, color='black', rotation=0, fontsize=20)
    ax2.tick_params(axis='y', labelleft='off', labelright='off', left=False, right=False)
    plt.show()

def graph_curv_2deriv():
    """
    Графіки порівняння курватур і другої похідної
    """
    FILE_PATH = file_arr[24][0]
    get_data_from_file_igor(FILE_PATH)
    EDC = [ar[240] for ar in data_dic["data"]]
    EDC = smooth_1D(EDC, 65)
    for i in range(608, 640):
        EDC[i] = EDC[i-1]

    Co = max(EDC)
    EDC =np.array(EDC)/Co
    
    grad_edc =np.gradient(EDC)
    grad2_edc =np.gradient(grad_edc)
    
    second_der = grad2_edc*(-1)
    second_der = smooth_1D(second_der, 15)
    second_der = second_der*(1/np.amax(second_der))

    curv1d = grad2_edc/(0.00001 + grad_edc**2)**(3 / 2)
    curv1d = curv1d*(-1)
    curv1d = smooth_1D(curv1d, 15)
    curv1d = curv1d*(1/np.amax(curv1d))
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    X=data_dic["y_scale"]
    ax.plot(X, EDC, color='black', linewidth=3, linestyle="-")
    ax.plot(X, second_der, color='grey', linewidth=3, linestyle="-.")
    ax.plot(X, curv1d, color='black', linewidth=3, linestyle="--")
    
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(20) 

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(20) 

    
    ax.set_xlabel('E, eV', fontsize=20)
    ax.set_ylabel('Інтенсивність', fontsize=20)
    plt.legend()
    plt.show()
    fig.savefig("spectr+curv_grad.png", dpi=300)

def spectr_vscurv():
    NUM = 7
    FILE_PATH = file_arr[NUM][0]
    FILE_DATA = file_arr[NUM]
    SHEAR = file_arr[NUM][4]
    POLAR = file_arr[NUM][5]
    get_data_from_file_igor(FILE_PATH)
    fermi_level, temp = DOS(shear=SHEAR)
    data_dic["x_scale"], data_dic["y_scale"] = axis_recalculate(fermi_level)
    #curv_plot_2x1()

    def x_axix_foo(data):
        xx = np.arange(0, len(data[0]), X_STEP)
        xlabels = [str(round(ax_x[i], 3)) for i in xx]
        return (xx, xlabels)

    def y_axix_foo(data):
        yy = np.arange(0, len(data), Y_STEP)
        ylabels = [str(round(ax_y[i], 3)) for i in yy]
        return (yy, ylabels)

    X_STEP = 80
    Y_STEP = 80
    ax_x = data_dic["x_scale"]
    ax_y = data_dic["y_scale"]
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax.set_title("Оригінальний спектр", color='black', fontsize=20)
    ax2.set_title("Оброблений методом кривизни", color='black', fontsize=20)
    ax.set_ylabel('E, eV', color='black', fontsize=20)
    ax.set_xlabel("k, 1/A", color='black', fontsize=20)
    ax2.set_xlabel("k, 1/A", color='black', fontsize=20)

    CV = curvature(93.0, 0.01, 15.0)
    im = ax.imshow(data_dic["data"], cmap="Greys")
    im = ax2.imshow(CV, cmap="Greys")
    

    xx, xlabels = x_axix_foo(data_dic["data"])
    yy, ylabels = y_axix_foo(data_dic["data"])
    ax.set_xticks(xx)
    ax.set_xticklabels(xlabels, color='black', rotation=0, fontsize=16)
    ax.set_yticks(yy)
    ax.set_yticklabels(ylabels, color='black', rotation=0, fontsize=20)    
    ax2.set_xticks(xx)
    ax2.set_xticklabels(xlabels, color='black', rotation=0, fontsize=16)    
    ax2.set_yticks(yy)
    ax2.set_yticklabels(ylabels, color='black', rotation=0, fontsize=20)
    ax2.tick_params(axis='y', labelleft='off', labelright='off', left=False, right=False)
    plt.show()

 
def many_curvature():
    arr_filter = {
        #("data_igor/FSS3_001.txt",)
        #"data_igor/FSS3_002.txt": {"beet": 240, "beet2": 340, "tup": (5.0, 0.001, 5.0)},
        "data_igor/FSS3_003.txt": {"beet": 240, "beet2": 340, "tup": (2.0, 0.001, 2.0)},    #36K
        "data_igor/FSS3_006.txt":{"beet": 200, "beet2": 340, "tup": (10.0, 0.001, 10.0)},   #27K
        #"data_igor/FSS3_007.txt": {"beet": 240, "beet2": 340, "tup": (5.0, 0.001, 5.0)},
        #"data_igor/FSS3_008.txt",
        #"data_igor/FSS3_009.txt",
        #"data_igor/FSS3_010.txt": {"beet": 200, "tup": (10.5, 0.001, 1.0)},
        #"data_igor/FSS3_011.txt",
        #"data_igor/FSS3_012.txt",
        #"data_igor/FSS3_013.txt": {"beet": 140, "tup": (20.5, 0.001, 1.0)},
        #"data_igor/FSS3_014.txt": {"beet": 140, "tup": (20.5, 0.001, 1.0)},
        #"data_igor/FSS3_015.txt",
        #"data_igor/FSS3_016.txt",
        #"data_igor/FSS3_017.txt",
        "data_igor/FSS3_018.txt": {"beet": 140, "tup": (20.5, 0.001, 1.0)},
        #"data_igor/FSS3_019.txt": {"beet": 140, "tup": (100.5, 0.001, 1.0)},
        #"data_igor/FSS3_020.txt",
        #"data_igor/FSS3_021.txt": {"beet": 180, "tup": (100.5, 0.001, 1.0)},
        #"data_igor/FSS3_022.txt",
        #"data_igor/FSS3_023.txt",
        #"data_igor/FSS3_024.txt",
        "data_igor/FSS3_025.txt": {"beet": 180, "tup": (2.5, 0.001, 10.0)},              # 77K
        #"data_igor/FSS3_026.txt": {"beet": 180, "tup": (80.5, 0.001, 10.0)},
        #"data_igor/FSS3_027.txt": {"beet": 180, "tup": (100.5, 0.001, 100.0)},
        #"data_igor/FSS3_028.txt": {"beet": 180, "tup": (50.5, 0.001, 50.0)},
        #"data_igor/FSS3_029.txt",
        #"data_igor/FSS3_030.txt",
        #"data_igor/FSS3_031.txt",
        #"data_igor/FSS3_032.txt",
        #"data_igor/FSS3_033.txt",
        #"data_igor/FSS3_034.txt",
        #"data_igor/FSS3_035.txt",
        #"data_igor/FSS3_036.txt",
        #"data_igor/FSS3_037.txt",
        #"data_igor/FSS3_038.txt",
        #"data_igor/FSS3_039.txt",
        #"data_igor/FSS3_040.txt",
        #"data_igor/FSS3_042.txt",
    }

    X_SIZE = data_dic["x_size"]
    Y_SIZE = data_dic["y_size"]
    
    array_cv = []
    for i in range(0, len(file_arr)):
        print("File path >", file_arr[i][0])
        if file_arr[i][0] not in arr_filter:
            continue
        global FILE_PATH
        global FILE_DATA
        FILE_PATH = file_arr[i][0]
        FILE_DATA = file_arr[i]
        SHEAR = file_arr[i][4]
        tp = arr_filter[FILE_PATH]["tup"]

        get_data_from_file_igor(FILE_PATH)
        fermi_level, temp = DOS(shear=SHEAR)

        CV = curvature(coef_cx=tp[0], coef_cy=tp[1], coef_i=tp[2]) 
        array_cv.append(CV)




    def x_axix_foo():
        xx = np.arange(0, X_SIZE, X_STEP)
        xlabels = [str(round(ax_x[i], 3)) for i in xx]
        return (xx, xlabels)

    def y_axix_foo():
        yy = np.arange(0, Y_SIZE, Y_STEP)
        ylabels = [str(round(ax_y[i], 3)) for i in yy]
        return (yy, ylabels)

    X_STEP = 100
    Y_STEP = 37
    data_dic["x_scale"], data_dic["y_scale"] = axis_recalculate(fermi_level) 
    ax_x = data_dic["x_scale"] 
    ax_y = data_dic["y_scale"]

    fig = plt.figure()
    ax1 = fig.add_subplot(141)
    ax2 = fig.add_subplot(142)
    ax3 = fig.add_subplot(143)
    ax4 = fig.add_subplot(144)

    ax1.set_title("27K", color='black', fontsize=16)
    ax2.set_title("36K", color='black', fontsize=16)
    ax3.set_title("56K", color='black', fontsize=16)
    ax4.set_title("77K", color='black', fontsize=16)

    ax1.set_ylabel('E, eV', color='black', fontsize=16)
    ax1.set_xlabel('k, 1/A', color='black', fontsize=16)
    ax2.set_xlabel('k, 1/A', color='black', fontsize=16)
    ax3.set_xlabel('k, 1/A', color='black', fontsize=16)
    ax4.set_xlabel('k, 1/A', color='black', fontsize=16)

    im = ax1.imshow(array_cv[0], cmap=cm.terrain)
    im = ax2.imshow(array_cv[1], cmap=cm.terrain)
    im = ax3.imshow(array_cv[2], cmap=cm.terrain)
    im = ax4.imshow(array_cv[3], cmap=cm.terrain)
    
    # Colorbar
    #
    #divider = make_axes_locatable(ax4)
    #cax = divider.append_axes('right', size='5%', pad=0.05)
    #fig.colorbar(im, cax=cax, orientation='vertical')
    
    
    xx, xlabels = x_axix_foo()
    yy, ylabels = y_axix_foo()
    
    ax1.set_xticks(xx)
    ax1.set_xticklabels(xlabels, color='black', rotation=0, fontsize=12)
    ax2.set_xticks(xx)
    ax2.set_xticklabels(xlabels, color='black', rotation=0, fontsize=12)
    ax3.set_xticks(xx)
    ax3.set_xticklabels(xlabels, color='black', rotation=0, fontsize=12)
    ax4.set_xticks(xx)
    ax4.set_xticklabels(xlabels, color='black', rotation=0, fontsize=12)

    ax1.set_yticks(yy)
    ax1.set_yticklabels(ylabels, color='black', rotation=0, fontsize=10)
    ax2.tick_params(axis='y', labelleft='off', labelright='off', left=False, right=False)
    ax3.tick_params(axis='y', labelleft='off', labelright='off', left=False, right=False)
    ax4.tick_params(axis='y', labelleft='off', labelright='off', left=False, right=False)
    #plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.tight_layout(w_pad=-0.9)
    plt.show()
    


GAUS_COEF = 6
MAX_LIMIT = 30  # curvature param ограничений части зануления шума
FILE_DATA = file_arr[0]
FILE_PATH = file_arr[0][0]
# get_data_from_file_igor(FILE_PATH)
#fermi_level, temp = DOS()


#x_arr = np.array(data_dic["x_scale"])
#y_arr = np.array(data_dic["y_scale"])
#z_arr = np.array(data_dic["data"])
# matrix of elements
#X, Y = scipy.meshgrid(x_arr, y_arr)
#Z = z_arr




# For recalculate data
#JSON = bias_for_temp()
#with open('data_FeSeTe.json', 'w') as fp:
#    json.dump(JSON, fp)

with open("data_FeSeTe.json") as fp:
    JSON = json.loads(fp.read())

#FOR debug
fermi_recalc(JSON)
chiterstvo(JSON)
"""
JSON = data_filter(JSON)
plot_fesete_debug(
    JSON=JSON,
    line_num=0,
    t_min=0,
    t_max=270,
    polar="all"
)
"""
#plot_fesete_2D()









# Функції для графіків
#
#graph_curv_2deriv()
spectr_vscurv()
#many_curvature()


























