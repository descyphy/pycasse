import os, sys, subprocess

dir_path = os.sep.join(sys.argv[0].split(os.sep)[:-1])
program_list = ['test_highway.py', 'test_merge.py', 'test_intersection.py']

for program in program_list:
    for freq in [2, 4, 8, 16]:
        for vehicle_num in range(2, 6):
            for cooperation in [False, True]:
                print('==================================================================================================')
                print('Running {} with {} {} vehicles and time step {} seconds.'.format(program, vehicle_num, 'cooperating' if cooperation else 'non-cooperating', 1/freq))
                subprocess.call(['python3', '{}{}{}'.format(dir_path, os.sep, program), str(vehicle_num), str(cooperation), str(freq)])