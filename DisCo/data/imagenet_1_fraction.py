import pandas as pd
import glob
import os
import numpy as np
import datetime
import sys
import random
import argparse
from shutil import copy


parser = argparse.ArgumentParser()
parser.add_argument('--ratio', type=int, default=1, help="img numbers to random")
args = parser.parse_args()

imagenet_dir = '/youtu-public/imagenet/ILSVRC/Data/CLS-LOC/train'
out_imagenet_dir = '/youtu-reid/jiaxzhuang/data/imagenet_{}_fraction'.format(args.ratio)

if not os.path.exists(out_imagenet_dir):
    os.mkdir(out_imagenet_dir)

if args.ratio == 1:
    file_txt = '1percent.txt'
elif args.ratio == 10:
    file_txt = '10percent.txt'
else:
    sys.exit(-1)
print('filename: ', file_txt)

txt = pd.read_csv(file_txt, delimiter='\t')
files_name = txt.to_numpy()[:, 0]

for num, name in enumerate(files_name):
    print(name)
    classes_name = name.split('_')[0]
    filename = os.path.join(imagenet_dir, classes_name)
    filename = os.path.join(filename, name)
    out_path = os.path.join(out_imagenet_dir, classes_name)
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    print('{}/{}=> copy {} to {}'.format(num, len(files_name), filename, out_path))
    copy(filename, out_path)

print('complete')
