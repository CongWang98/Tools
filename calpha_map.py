import os

input_file = 'input/traj_center.lammpstrj'
output_file = 'output/traj_center_ca.lammpstrj'


fin = open(input_file, 'r')
fout = open(output_file, 'w')
frame = 0
while True:
    line = fin.readline()
    if not line:
        break
    if (line + ' 1').split()[1] == 'ATOMS':
        fout.write(line)
        frame += 1
        for i in range(132):
            line = fin.readline()
            if (i % 10 == 8) & ((i//10)+1<=12):
                coorforce_lis = line.split()[2:]
                fout.write('{} {} '.format((i//10)+1, 1))
                fout.write(' '.join(coorforce_lis))
                fout.write('\n')
            else:
                pass
        if frame%10000 == 0:
            print ' {} frames mapped.'.format(frame)
    elif (line + ' 1').split()[1] == 'NUMBER':
        fout.write(line)
        fin.readline()
        fout.write('12\n')
    else:
        fout.write(line)

fin.close()
fout.close()
