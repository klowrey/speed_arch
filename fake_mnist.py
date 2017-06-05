
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--out_file')
args = parser.parse_args()
out_file = args.out_file

time.sleep(0.5)

TEST_DATA = {'speed': 0.05, 'accuracy': 0.93, 'nparams':500}

with open(out_file, 'w') as f:
    f.write(str(TEST_DATA))
