import sys
import os

if __name__ == '__main__':
    for i in range(20):
        line = input()
        score = int(line.split(' ')[0].split('\t')[1]) - 10000000000
        print(score)