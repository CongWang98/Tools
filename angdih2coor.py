import math
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def get_ang(a, b, c):
    '''
    The angle is a-b-c. Return a scale value ranging from 0-1(unit: pi).
    '''
    # print('a=', a)
    # print('b=', b)
    # print('c=', c)
    ba = np.array(a) - np.array(b)
    # print('ba=', ba)
    bc = np.array(c) - np.array(b)
    # print('bc=', bc)
    cos = np.dot(ba, bc).sum() / (get_abs(ba) * get_abs(bc))
    # print('ba*bc/|ba||bc|=', cos)
    # print('theta=', math.acos(cos) * 180 / math.pi)
    return math.acos(cos) / math.pi


def get_dih(a, b, c, d):
    '''
    The dihedral is a-b-c-d.Return a scale value ranging from 0-1(unit: 2pi).
    '''
    # print('a=', a)
    # print('b=', b)
    # print('c=', c)
    # print('d=', d)
    n1 = np.cross(np.array(a) - np.array(b), np.array(c) - np.array(b))
    # print('n1=ba cross bc=', n1)
    n2 = np.cross(np.array(b) - np.array(c), np.array(d) - np.array(c))
    # print('n2=cb cross cd=', n2)
    cos = np.dot(n1, n2).sum() / (get_abs(n1) * get_abs(n2))
    # print('cos=n1*n2/|n1||n2|=', cos)
    # print('theta=', math.acos(cos) * 180 / math.pi)
    theta = math.acos(cos)
    if np.dot(n1, np.array(d) - np.array(c)) < 0:
        theta = 2 * math.pi - theta
    return theta / (2 * math.pi)


def get_dis(a, b):
    '''
    Return the distance between two atom.
    '''
    tmp = np.array(a) - np.array(b)
    return math.sqrt(np.dot(tmp, tmp).sum())


def get_abs(a):
    '''
    Return the absolute value of a verctor.
    '''
    return get_dis(a, [0, 0, 0])


def GetCoor(a, b, c, angle, dihedral, bondlength):
    """
    The sequance is a-b-c-d. Give a angle(0-1, unit:pi), a dihedral(0-1, unit:2*pi) and the coordinate(unit: A) of a, b, c,
    return the coordinate of d.
    """
    ang = angle * math.pi
    dih = dihedral * 2 * math.pi
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    n1 = np.cross(ba, bc)
    cb_yan = bc / get_abs(bc) * (get_abs(ba) * math.cos(get_ang(a, b, c) * math.pi))
    ha = ba - cb_yan
    bc_yan = - bc / get_abs(bc) * (bondlength * math.cos(ang))
    hd = n1 / get_abs(n1) * (bondlength * math.sin(ang) * math.sin(dih))
    hh = ha / get_abs(ha) * (bondlength * math.sin(ang) * math.cos(dih))
    return c + bc_yan + hh + hd


def CalBondLength(coor_frame):
    """
    Give a frame of CGed trajectory, return the 'bond' length.
    """
    atom_lis = coor_frame.reshape(-1, 3)
    length_lis = []
    for i in range(len(atom_lis) - 1):
        atom1 = atom_lis[i]
        atom2 = atom_lis[i + 1]
        bond_length = (((atom2 - atom1) ** 2).sum()) ** (1 / 2)
        length_lis.append(bond_length)
    return np.array(length_lis)


def CalAveBondLength(coorlis):
    """
    Give a coorlis, return the average bond length list.
    """
    bond_lis_lis = np.array([CalBondLength(i) for i in coorlis])
    return bond_lis_lis.mean(0)


def AngdihToCoor(angdihlis, bondlength_lis):
    """
    Convert to a angdihlis to coordination list.
    """
    atom_num = int((len(angdihlis) + 5) / 2)
    if atom_num != len(bondlength_lis) + 1:
        raise ValueError('Inconsistant atom number')
    coor_lis = []
    angle = angdihlis[0]
    a = [0, 0, 0]
    b = [0, 0, bondlength_lis[0]]
    c = [0, bondlength_lis[1] * math.sin(angle * math.pi), bondlength_lis[0] - bondlength_lis[1] * math.cos(angle * math.pi)]
    coor_lis = np.array(coor_lis + a + b + c)
    for i in range(atom_num - 3):
        dihedral, angle = angdihlis[2 * i + 1], angdihlis[2 * i + 2]
        a, b, c = coor_lis[3 * i: 3 * i + 3], coor_lis[3 * i + 3: 3 * i + 6], coor_lis[3 * i + 6: 3 * i + 9]
        bondlength = bondlength_lis[i + 2]
        d = GetCoor(a, b, c, angle, dihedral, bondlength)
        coor_lis = np.concatenate((coor_lis, d), axis=0)
    return np.array(coor_lis)


def GenePdbString(coor_frame, CAlis=None):
    """
    Give a coor_frame , return pdb file string.
    """
    pdbs = 'CRYST1   60.000   60.000   60.000  90.00  90.00  90.00 P 1           1\n'
    atomlis = coor_frame.reshape(-1, 3)
    if not CAlis:
        CAlis = ['ALA'] * len(atomlis)
    elif len(CAlis) != len(atomlis):
        raise IndexError('Length of CA list not equal to length of CA index lis')
    for i in range(len(atomlis)):
        pdbs += 'ATOM    {:>3d}  CA  {} A  {:>2d}     {:7.3f} {:7.3f} {:7.3f}  1.00  0.00           C\n'.format(i + 1, CAlis[i], i + 1, atomlis[i][0], atomlis[i][1], atomlis[i][2])
    pdbs += 'END\n'
    return pdbs


def CoorlisToLmpstrj(coorlis, outfilepath, boxlength=60):
    """
    Generate lammps trajectory based on the given coorlis.
    """
    frame = len(coorlis)
    if len(coorlis[0]) % 3 != 0:
        raise ValueError('length of coorlis % 3 != 0')
    atom = int(len(coorlis[0]) / 3)
    f = open(outfilepath)
    for i in range(frame):
        coor_frame = coorlis[i]
        minx = min([coor_frame[j] for j in range(0, len(coor_frame), 3)])
        miny = min([coor_frame[j] for j in range(1, len(coor_frame), 3)])
        minz = min([coor_frame[j] for j in range(0, len(coor_frame), 3)])
        f.write('ITEM: TIMESTEP\n{}\n'.format(i))
        f.write('ITEM: NUMBER OF ATOMS\n{}\n'.format(atom))
        f.write('ITEM: BOX BOUNDS pp pp pp\n{} {}\n{} {}\n{} {}'.format(minx,
                minx + boxlength, miny, miny + boxlength,
                minz, minz + boxlength))
        f.write('ITEM: ATOMS id type xu yu zu\n')
        for j in range(atom):
            x_, y_, z_ = coor_frame[3 * j], coor_frame[3 * j + 1], coor_frame[3 * j + 2]
            f.write('{}\t{}\t{}\t{}\t{}\n'.format(j + 1, 1, x_, y_, z_))
    f.close()
    print('[INFO] Re-build a trj file {} from a coordinate list'.format(outfilepath))


def ConformationPlot(coor_frame):
    """
    Give a frame of trajectory, plot it.
    """
    x_lis = [coor_frame[j] for j in range(0, len(coor_frame), 3)]
    y_lis = [coor_frame[j] for j in range(1, len(coor_frame), 3)]
    z_lis = [coor_frame[j] for j in range(2, len(coor_frame), 3)]
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x_lis, y_lis, z_lis, c='r', s=30)
    ax.set_zlabel('z')
    ax.set_ylabel('y')
    ax.set_xlabel('x')
    plt.show()
