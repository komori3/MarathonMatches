import sys
import os
import shutil
from subprocess import check_output

testcase_generator = os.path.join('testcase_generator.cpp')
testcase_master_list = os.path.join('testcase_master_list.txt')
testcase_list = os.path.join('testcase_list.csv')

def complie_testcase_generator():
    cmd = ['g++', '-O2', testcase_generator, '-o', testcase_generator.split('.')[0]]
    check_output(cmd).decode()

def generate_testcases():
    with open(testcase_master_list) as f:
        data = str(f.read())
    
    lines = []
    for line in data.split('\n'):
        if line == '':
            continue
        lines.append(line)

    print(lines)
    
    test_dir = os.path.join('testcase')
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)

    with open(os.path.join(test_dir, testcase_list), 'w') as tlist_file:
        tlist_file.write('group,seed,path\n')

        for group_no in range(len(lines)):
            line = lines[group_no]
            test_sub_dir = os.path.join(test_dir, str(group_no))
            os.makedirs(test_sub_dir)

            params = line.split(' ')
            print(params)
            for seed in range(1, 11):
                cmd = ['./' + testcase_generator.split('.')[0]]
                cmd += params
                cmd += ['-Seed', str(seed)]
                print(cmd)
                out = check_output(cmd).decode()
                test_file = os.path.join(test_sub_dir, str(seed) + '.txt')
                with open(test_file, 'w') as f:
                    f.write(out)
                tlist_file.write(str(group_no) + ',' + str(seed) + ',' + os.path.abspath(test_file) + '\n')

def destroy_testcase_generator():
    os.remove(os.path.join(testcase_generator.split('.')[0]))

if __name__ == '__main__':

    complie_testcase_generator()
    generate_testcases()
    destroy_testcase_generator()


