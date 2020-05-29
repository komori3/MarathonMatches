import os
import sys
import json
import re
import shutil
import subprocess
from subprocess import check_output
from operator import itemgetter
from datetime import datetime
from multiprocessing import Pool
import argparse
import time
import random
import string
import yaml

def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('seeds', type=str)
  parser.add_argument('out_folder', type=str)
  args = parser.parse_args()
  return args

def exec(cmd):
    sc = cmd.split(' ')

    elapsed = time.perf_counter()
    out = check_output(sc).decode()
    outs = out.split('\n')
    elapsed = time.perf_counter() - elapsed

    sp = [s for s in re.split(r'[ \n\r]', outs[0]) if s != '']
    score = float(sp[-1])
    sp = [s for s in re.split(r'[ \n\r]', outs[1]) if s != '']
    seed = int(sc[-1])
    print('seed = ' + str(seed) + ', score = ' + str(score))
    return {'seed': seed, 'score': score}


if __name__ == "__main__":
  args = get_args()
  seeds_path = os.path.join(args.seeds)
  if not os.path.exists(seeds_path):
    assert False
  out_folder = os.path.join(args.out_folder)
  if os.path.exists(out_folder):
    assert False
  
  num_procs = 10
  with open(seeds_path, 'r', encoding='utf-8') as seeds_file:
    seeds_str = str(seeds_file.read())
    seeds = seeds_str.split('\n')

  build_cmd = 'g++ -std=gnu++11 -O3 ../src/DanceFloor.cpp -o DanceFloor'
  print('now compiling...')
  subprocess.run(build_cmd, shell=True)
  print('compile succeeded.')

  cmd_list = []
  for seed in seeds[:10]:
    if seed == '':
      continue
    tester_cmd = f'java -jar ../bin/tester.jar -ex ./DanceFloor -nv -no -so {out_folder} -th 10 -sd {seed}'
    cmd_list.append(tester_cmd)

  print(cmd_list)
  print('num_procs: ' + str(num_procs))
  with Pool(processes=num_procs) as pool:
      results = pool.map(exec, cmd_list)

  with open(os.path.join(out_folder, 'results.csv'), 'w', encoding='utf-8') as f:
    for result in results:
      f.write(str(result['seed']) + ',' + str(result['score']) + '\n')
    