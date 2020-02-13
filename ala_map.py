#!/usr/bin/env python2
# envirment: cgmap(Anaconda2/4.3.0)
# Run on midway2 

import sys
sys.path.append('/project/gavoth/congwang/cgmap/src/')
sys.path.append('/project/gavoth/congwang/mdtraj/mdtraj/')
import cgmap as cg
import mdtraj as md
import md_check as check

############################### config #####################################

input_traj = 'helix.trr'
input_top  = 'helix.gro'

output_traj = 'helix_cg.lammpstrj'
output_top  = 'helix_cg.gro'

output_dir = './output/'
input_dir  = './input/'

############################### run ########################################

### pull in trajectories
trj = md.load(input_dir + input_traj, top=input_dir + input_top)

### define mapping based on knowledge of topology
### in this instance, map every residue into a single site
for a in trj.top.atoms: 
    a.mass = a.element.mass
    a.charge = 0

# first residue is ACE1 (zero index'd)
name_lists = []
label_lists = []
molecule_types = []
resREF = 1
istart = 0
iend = 0
iname = 'ACE'
molnum = 0

maxSize = len(list(trj.top.atoms))
tempMol = []
tempCGL = []
for i, a in enumerate(trj.top.atoms) :
    resNAME = str(a.residue)[0:3]
    print resNAME
    resNUM = int(str(a.residue)[3:])
    print resNUM
    aINDEX = a.index
    print aINDEX

    if (resNUM != resREF):
	resREF = resNUM
	if resNUM == 2:
	    pass
	elif resNUM == 14:
	    iname = 'NAC'
	else: 
            #first append name_lists and label
            iend = aINDEX - 1
            tempMol.append("index %d to %d" % (istart, iend))
            tempCGL.append(iname)

            #then update things for next residue
            iname = resNAME
            istart = aINDEX

    # special case if last item
    if (i == (maxSize-1)) :
        iend = aINDEX
        tempMol.append("index %d to %d" % (istart, iend))
        tempCGL.append(iname)
        molecule_types.append(int(molnum))
        name_lists.append(tempMol)
        label_lists.append(tempCGL)

###actual map command
print name_lists
print label_lists
print molecule_types

print "Lengths of all three lists should be equivalent: %d = %d = %d" % (len(name_lists), len(label_lists), len(molecule_types))

cg_trj = cg.map_unfiltered_molecules(            trj = trj,
                                      selection_list = name_lists,
                                     bead_label_list = label_lists,
                                      molecule_types = molecule_types,
                                      mapping_function = "com")
#print trj.forces[0]
#print cg_trj.forces[0]
cg_trj.save_lammpstrj(output_dir + output_traj)
cg_trj[0].save(output_dir + output_top)
