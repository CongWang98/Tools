import mdtraj as md
import math
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt

def dis_cal(a, b):
    dx_2 = (a[0] - b[0]) ** 2
    dy_2 = (a[1] - b[1]) ** 2
    dz_2 = (a[2] - b[2]) ** 2
    dis = math.sqrt(dx_2 + dy_2 + dz_2)
    #print dis 
    return dis

def Q_cal(lis):
    N = len(lis)
    sig2 = 0.02
    R_0 = 0.5
    sum = 0
    for i in lis:
        #print i
        sum += math.exp(-(1 / (2 * sig2)) * (i - R_0) ** 2)
    return sum / N


def get_dis(trj, index):
    dis = []
    for i in range(9):
        a = trj.xyz[index][i]
        b = trj.xyz[index][i + 3]
        #print (a, b)
	dis.append(dis_cal(a, b))
    return dis



def get_Q_lis(trj):
    Q_lis = []
    for i in tqdm(range(len(trj))):
        dis = get_dis(trj, i)
        Q_lis.append(Q_cal(dis))
    #print Q_lis
    #print len(Q_lis)
    return Q_lis


def get_rg(filename):
    rg_lis = []
    with open(filename, 'r') as f:
        while True:	
            line = f.readline()
	    if not line:
		break
	    if line[0]!='@' and line[0] != '#':
		rg_lis.append(float(line.split()[1]))
    return rg_lis

def get_rmsd(filename):
    rmsd_lis = []
    with open(filename, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            if line[0]!='@' and line[0] != '#':
                rmsd_lis.append(float(line.split()[1]))
    return rmsd_lis



def write_Rg_Qhel(trj, rgfile, outfilename):
    Q_lis = get_Q_lis(trj)
    rg_lis = get_rg(rgfile)
    if len(Q_lis) != len(rg_lis):
	raise ValueError ('The length of Q_lis != the length ofrg_lis')
    with open(outfilename, 'w') as f:
	f.write('Rg    Qhel\n')
	for i in tqdm(range(len(Q_lis))):
	    f.write('{}    {}\n'.format(rg_lis[i],Q_lis[i]))


def mat_his(rg_qhel_file, solution, imgfile):
    xVals = []
    yVals = []
    with open(rg_qhel_file, 'r') as f:
        line = f.readline()
        while True:
            line = f.readline()
            if not line:
                break
            rg, qhel = line.split()
            xVals.append(float(rg))
            yVals.append(float(qhel))
    #plt.plot(xVals, yVals, '.r')
    #plt.show()
    plt.hist2d(xVals, yVals,range=[[0.4,1.2],[0,1]], bins=(solution, solution), cmap=plt.cm.jet)
    plt.xlabel('Radius of gyration (nm)')
    plt.ylabel('$Q_{hel}$')
    

    plt.savefig(imgfile,dpi=1000)

def mat_his_rmsd(rg_rmsd_file, solution, imgfile):
    xVals = []
    yVals = []
    with open(rg_rmsd_file, 'r') as f:
        line = f.readline()
        while True:
            line = f.readline()
            if not line:
                break
            rg, rmsd = line.split()
            xVals.append(float(rg))
            yVals.append(float(rmsd))
    #plt.plot(xVals, yVals, '.r')
    #plt.show()
    plt.hist2d(xVals, yVals,range=[[0.4,1.2],[0,0.8]], bins=(solution, solution), cmap=plt.cm.jet)
    plt.xlabel('Radius of gyration (nm)')
    plt.ylabel('$RMSD (nm)$')
    plt.savefig(imgfile,dpi=1000)


def cal_rg(trj, outfile):
    mass_lis = np.array([12.011,12.011 , 12.011, 12.011, 12.011, 12.011, 12.011, 12.011, 12.011, 12.011, 12.011, 12.011])
    rg_lis =  md.compute_rg(trj, mass_lis)
    with open(outfile, 'w') as f:
        for i in range(len(rg_lis)):
            f.write('{}\t{}\n'.format(i + 1, rg_lis[i]))

def cal_rmsd(trj, rmsd_reftrj, outfile):
    rmsd_lis = md.rmsd(trj, rmsd_reftrj)
    with open(outfile, 'w') as f:
        for i in range(len(rmsd_lis)):
            f.write('{}\t{}\n'.format(i + 1, rmsd_lis[i]))


def write_Rg_rmsd(rgfile,rmsdfile, outfilename):
    rg_lis = get_rg(rgfile)
    rmsd_lis = get_rmsd(rmsdfile)
    if len(rmsd_lis) != len(rg_lis):
        raise ValueError ('The length of rmsd_lis != the length of rg_lis')
    with open(outfilename, 'w') as f:
        f.write('Rg    Qhel\n')
        for i in tqdm(range(len(rg_lis))):
            f.write('{}    {}\n'.format(rg_lis[i],rmsd_lis[i]))



def histogram2dGnuplot(rg_qhel_file, xedges, yedges, outfile, allowOutsideValues=False, normed=False, freeEnergy=False, kbt=-1):
    xVals = []
    yVals = []
    with open(rg_qhel_file, 'r') as f:
	line = f.readline()
	while True:
	    line = f.readline()
	    if not line:
		break
	    rg, qhel = line.split()
	    xVals.append(float(rg))
	    yVals.append(float(qhel))	
    hist2d = np.zeros((len(yedges) - 1, len(xedges) - 1))
    numVals = len(xVals)
    if freeEnergy == True:
        normed = True
        if kbt < 0:
            raise ValueError('kbt not specified')
    if numVals != len(yVals):
        raise ValueError('Length of x and y arrays are not the same')
    for i in range(numVals):
        xindex = -1
        yindex = -1
        for j in range(len(xedges) - 1):
            if xVals[i] > xedges[j] and xVals[i] <= xedges[j+1]:
                xindex = j
        for j in range(len(yedges) - 1):
            if yVals[i] > yedges[j] and yVals[i] <= yedges[j+1]:
                yindex = j
        if xindex < 0 and not allowOutsideValues:
            print xVals[i]
            raise ValueError('x value outside grid!')
        if yindex < 0 and not allowOutsideValues:
            print yVals[i]
            raise ValueError('y value outside grid!')
        hist2d[yindex][xindex] += 1
    with open(outfile, 'w') as f:
    	for i, yArray in enumerate(hist2d):
            ycenter = (yedges[i] + yedges[i+1]) / 2.0
            ywidth = (yedges[i+1] - yedges[i]) / 2.0
            for j, val in enumerate(yArray):
                xcenter = (xedges[j] + xedges[j+1]) / 2.0
                xwidth = (xedges[j+1] - xedges[j]) / 2.0
                outVal = val
                if normed:
                    outVal = val / (float(numVals) * xwidth * ywidth)
                if freeEnergy:
		    if outVal == 0:
			outVal = 1e-12
                    outVal = -1*kbt*np.log(outVal)
                f.write('%10.6f %10.6f %10.6f\n' % (xcenter, ycenter, outVal))
            f.write('\n')



if __name__ == '__main__':
    
    trjfile = 'traj_center_ca_1w.lammpstrj'
    topfile = 'traj_center_cg.gro'	
    rmsd_reftrjfile = 'helix_cg.lammpstrj'
    rmsd_reftopfile = 'helix_cg.gro'
   
    outfile = 'rg_qhel/rg_qhel_{}.dat'.format(trjfile)
    outfile_his = 'rg_qhel/rg_qhel_his_{}.dat'.format(trjfile)
    imgfile = 'img/Qhel_Rg_{}.png'.format(trjfile)
    rg_file = 'rg/rg_{}.xvg'.format(trjfile)
   
    outfile1 = 'rg_rmsd/rg_rmsd_{}.dat'.format(trjfile)
    outfile_his1 = 'rg_rmsd/rg_rmsd_his_{}.dat'.format(trjfile)
    imgfile1 = 'img/Rmsd_Rg_{}.png'.format(trjfile)
    rmsd_file = 'rmsd/rmsd_{}.xvg'.format(trjfile)

    
    trj = md.load(trjfile, top=topfile)
    print trj
    rmsd_reftrj = md.load(rmsd_reftrjfile, top=rmsd_reftopfile)
    print rmsd_reftrj
    cal_rg(trj, rg_file)
    cal_rmsd(trj, rmsd_reftrj, rmsd_file)
    write_Rg_Qhel(trj, rg_file, outfile)
    write_Rg_rmsd(rg_file,rmsd_file, outfile1)
    # histogram2dGnuplot(outfile, xedges, yedges, outfile_his, normed=True, freeEnergy=True, kbt=2.577, allowOutsideValues=False)
    mat_his(outfile, 200, imgfile)
    mat_his_rmsd(outfile1, 200, imgfile1)






