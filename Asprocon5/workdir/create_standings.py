import sys
import os
import shutil
import argparse
from subprocess import check_output

def get_args():
  usage = 'python {} --submissions SUBMISSIONS_DIR [--group GROUP_SET]'.format(__file__)
  parser = argparse.ArgumentParser(usage=usage)
  parser.add_argument('-s', '--submissions', help='submissions directory', type=str, required=True)
  parser.add_argument('-g', '--group', help='group set (ex. 1,2,3)', type=str)
  args = parser.parse_args()
  return args

def get_num_groups():
  lines = []
  with open('testcase_master_list.txt') as f:
    data = str(f.read())
    for line in data.split('\n'):
      if line == '':
        continue
      lines.append(line)
  return len(lines)

if __name__ == '__main__':
  args = get_args()

  groupset = []
  if args.group:
    for sg in args.group.split(','):
      groupset.append(int(sg))
  else:
    for g in range(get_num_groups()):
      groupset.append(g)

  print(groupset)

  submissions_dir = os.path.join(args.submissions)

  if not os.path.exists(submissions_dir):
    print('directory {} does not exist.'.format(os.path.abspath(submissions_dir)))
    exit()
  
  submission_dirs = []
  dict_submission_to_scores = {}
  dict_submission_to_total_score = {}

  for submission in os.listdir(submissions_dir):
    submission_dir = os.path.join(submissions_dir, submission)
    submission_dirs.append(submission_dir)
    dict_submission_to_scores[submission] = {}

  seed_set = set()

  for submission_dir in submission_dirs:
    submission_name = submission_dir.split('/')[-1]
    dict_submission_to_total_score[submission_name] = 0

    results_csv = os.path.join(submission_dir, 'results.csv')
    lines = []
    with open(results_csv) as f:
      data = str(f.read())
      for line in data.split('\n'):
        if line == '':
          continue
        lines.append(line)
    for line in lines[1:]:
      cols = line.split(',')
      group_no = int(cols[0])
      if not group_no in groupset:
        continue
      seed = (int(cols[0]), int(cols[1]))
      seed_set.add(seed)
      dict_submission_to_scores[submission_name][seed] = int(cols[3])
  
  for seed in seed_set:
    max_score = 0
    dict_submission_to_seed_score = {}
    for submission_name, submission_scores in dict_submission_to_scores.items():
      dict_submission_to_seed_score[submission_name] = submission_scores[seed]
      max_score = max(max_score, submission_scores[seed])
    if max_score == 0:
      continue
    for submission, score in dict_submission_to_seed_score.items():
      dict_submission_to_seed_score[submission] = score / max_score
      dict_submission_to_total_score[submission] += score / max_score

  max_length = len('submission')
  list_total_score_to_submission = []
  for submission, total_score in dict_submission_to_total_score.items():
    max_length = max(max_length, len(submission))
    list_total_score_to_submission.append((total_score, submission))

  list_total_score_to_submission.sort()

  space = max_length - len('submission') + 4
  print('submission' + (' ' * space) + 'score')
  print('-' * 30)
  for total_score, submission in list_total_score_to_submission:
    space = max_length - len(submission) + 4
    print(submission + (' ' * space) + str(total_score))

