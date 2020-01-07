import os
import sys
import json
import re
import shutil
import subprocess
#from subprocess import check_output
from operator import itemgetter
from datetime import datetime
from multiprocessing import Pool
import sqlite3
import argparse
import time
import random
import string

def random_name(n):
   return ''.join(random.choices(string.ascii_letters + string.digits, k=n))

def get_args():
    usage = 'python {} --tester TESTER --exec EXEC --source SOURCE --testcase TESTCASE_DIR --target TARGET_DIR [--j <num_procs>]'.format(__file__)
    parser = argparse.ArgumentParser(usage=usage)
    parser.add_argument('--tester', help='tester commands (ex. java -jar tester.jar -novis)', type=str, required=True)
    parser.add_argument('--exec', help='exec file', type=str, required=True)
    parser.add_argument('--source', help='source file', type=str, required=True)
    parser.add_argument('--testcase', help='testcase directory', type=str, required=True)
    parser.add_argument('--target', help='target directory', type=str, required=True)
    parser.add_argument('--j', help='num processors', type=int, default=1)
    args = parser.parse_args()
    return args

def exec(param):
    group_no = param['group_no']
    seed = param['seed']
    tester = param['tester']
    solver = param['solver']
    testcase = param['testcase']
    target = param['target']

    cmd = [solver]

    with open(testcase) as f:
        input = f.read()
        input_byte = bytes(input.encode())
    
    elapsed = time.perf_counter()
    ret = subprocess.run(cmd, input=input_byte, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    elapsed = time.perf_counter() - elapsed

    output = ret.stdout.decode('utf8')

    tmp_out = os.path.join(target, random_name(16) + '.txt')

    with open(tmp_out, 'w') as f:
        f.write(output)
    
    cmd2 = [tester, testcase, tmp_out]
    ret2 = subprocess.run(cmd2, stdout=subprocess.PIPE)

    os.remove(tmp_out)

    line = group_no + ',' + seed + ',' + testcase + ',' + ret2.stdout.decode('utf8')[:-1] + ',' + str(int(elapsed * 1000.0))

    print(line)

    return line

def systest(tester, solver, source, testcases, target, num_procs):
    
    os.makedirs(target)

    with open(testcases) as f:
        data = str(f.read())
    
    lines = []
    for line in data.split('\n'):
        if line == '':
            continue
        lines.append(line)

    param_list = []
    for line in lines[1:]:
        param = {}
        group_no, seed, testcase = line.split(',')
        param['group_no'] = group_no
        param['seed'] = seed
        param['testcase'] = testcase
        param['tester'] = tester
        param['solver'] = solver
        param['target'] = target
        param_list.append(param)

    with Pool(processes = num_procs) as pool:
        results = pool.map(exec, param_list)    

    print(source, os.path.join(target, source))
    shutil.copy(source, os.path.join(target, source.split('/')[-1]))

    results_csv = os.path.join(target, 'results.csv')
    with open(results_csv, 'w') as f:
        f.write('group_no,seed,path,score,elapsedMs\n')
        for result in results:
            f.write(result + '\n')

if __name__ == "__main__":

    # param = ['./output_checker', '../asprocon5', './testcase/0/1.txt']
    # exec(param)

    # exit(1)

    args = get_args()

    tester = args.tester
    solver = args.exec
    source = args.source
    testcase = args.testcase
    target = args.target
    num_procs = args.j

    if not os.path.exists(tester):
        print('failed: open ' + tester)
        exit()
    if not os.path.exists(solver):
        print('failed: open ' + solver)
        exit()
    if not os.path.exists(source):
        print('failed: open ' + source)
        exit()
    if not os.path.exists(testcase):
        print('failed: open {}'.format(os.path.abspath(testcase)))
        exit()
    if os.path.exists(target):
        print('target directory {} already exists.'.format(os.path.abspath(target)))
        exit()

    systest(tester, solver, source, testcase, target, num_procs)