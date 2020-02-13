import os

input_file = 'output/traj_center_ca_mod.lammpstrj'
output_file = 'output/traj_center_ca_mod_1w.lammpstrj'


fin = open(input_file, 'r')
fout = open(output_file, 'w')
line = fin.readline()
while True:
    if not line:
        break
    if (line + ' 1 1').split()[1] == 'TIMESTEP':
        line = fin.readline()
        timestep = int(line.split()[0])
        if timestep % 55 == 0:
            print('\r {}'.format(timestep//55), end='')
            fout.write('ITEM: TIMESTEP\n{}\n'.format(timestep//55))
            line = fin.readline()
            while (line + ' 1 1').split()[1] != 'TIMESTEP':
                fout.write(line)
                line = fin.readline()
        else:
            pass
    else:
        line = fin.readline()
fin.close()
fout.close()
