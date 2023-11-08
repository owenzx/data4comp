import numpy as np
import argparse
import os


ROOT_FOLDER = os.path.dirname(os.path.realpath(__file__))



def generate_cogs_good_dev(dev_portion=0.1):
    cogs_raw_folder = os.path.join(ROOT_FOLDER, 'COGS/cogs')
    gen_file = os.path.join(cogs_raw_folder, 'gen.tsv')
    with open(gen_file, 'r') as fr:
        lines = fr.readlines()
    rand_perm = np.random.permutation(len(lines))
    dev_size = int(len(lines) * dev_portion)
    dev_ids = rand_perm[:dev_size]
    new_test_ids = rand_perm[dev_size:]
    dev_lines = [lines[i] for i in dev_ids]
    new_test_lines = [lines[i] for i in new_test_ids]

    good_dev_file = os.path.join(cogs_raw_folder, 'dev_gen.tsv')
    new_test_file = os.path.join(cogs_raw_folder, 'new_test.tsv')

    with open(good_dev_file, 'w') as fw:
        for line in dev_lines:
            fw.write(line)

    with open(new_test_file, 'w') as fw:
        for line in new_test_lines:
            fw.write(line)



def generate_scan_addjump_good_dev(dev_portion=0.1):
    scan_raw_folder = os.path.join(ROOT_FOLDER, 'SCAN/add_prim_split')
    gen_file = os.path.join(scan_raw_folder, 'tasks_test_addprim_jump.txt')

    with open(gen_file, 'r') as fr:
        lines = fr.readlines()
    rand_perm = np.random.permutation(len(lines))
    dev_size = int(len(lines) * dev_portion)
    dev_ids = rand_perm[:dev_size]
    new_test_ids = rand_perm[dev_size:]
    dev_lines = [lines[i] for i in dev_ids]
    new_test_lines = [lines[i] for i in new_test_ids]

    good_dev_file = os.path.join(scan_raw_folder, 'tasks_gooddev_addprim_jump.txt')
    new_test_file = os.path.join(scan_raw_folder, 'tasks_newtest_addprim_jump.txt')

    with open(good_dev_file, 'w') as fw:
        for line in dev_lines:
            fw.write(line)

    with open(new_test_file, 'w') as fw:
        for line in new_test_lines:
            fw.write(line)



def generate_scan_adddax_good_dev(dev_portion=0.1):
    scan_raw_folder = os.path.join(ROOT_FOLDER, 'SCAN/add_prim_split')
    gen_file = os.path.join(scan_raw_folder, 'tasks_test_addprim_dax.txt')

    with open(gen_file, 'r') as fr:
        lines = fr.readlines()
    rand_perm = np.random.permutation(len(lines))
    dev_size = int(len(lines) * dev_portion)
    dev_ids = rand_perm[:dev_size]
    new_test_ids = rand_perm[dev_size:]
    dev_lines = [lines[i] for i in dev_ids]
    new_test_lines = [lines[i] for i in new_test_ids]

    good_dev_file = os.path.join(scan_raw_folder, 'tasks_gooddev_addprim_dax.txt')
    new_test_file = os.path.join(scan_raw_folder, 'tasks_newtest_addprim_dax.txt')

    with open(good_dev_file, 'w') as fw:
        for line in dev_lines:
            fw.write(line)

    with open(new_test_file, 'w') as fw:
        for line in new_test_lines:
            fw.write(line)



def generate_scan_addaroundright_good_dev(dev_portion=0.1):
    scan_raw_folder = os.path.join(ROOT_FOLDER, 'SCAN/template_split')
    gen_file = os.path.join(scan_raw_folder, 'tasks_test_template_around_right.txt')

    with open(gen_file, 'r') as fr:
        lines = fr.readlines()
    rand_perm = np.random.permutation(len(lines))
    dev_size = int(len(lines) * dev_portion)
    dev_ids = rand_perm[:dev_size]
    new_test_ids = rand_perm[dev_size:]
    dev_lines = [lines[i] for i in dev_ids]
    new_test_lines = [lines[i] for i in new_test_ids]

    good_dev_file = os.path.join(scan_raw_folder, 'tasks_gooddev_template_around_right.txt')
    new_test_file = os.path.join(scan_raw_folder, 'tasks_newtest_template_around_right.txt')

    with open(good_dev_file, 'w') as fw:
        for line in dev_lines:
            fw.write(line)

    with open(new_test_file, 'w') as fw:
        for line in new_test_lines:
            fw.write(line)




def generate_scan_mcd1_good_dev(dev_portion=0.1):
    scan_raw_folder = os.path.join(ROOT_FOLDER, 'SCAN/mcd_split')
    gen_file = os.path.join(scan_raw_folder, 'tasks_test_mcd1.txt')

    with open(gen_file, 'r') as fr:
        lines = fr.readlines()
    rand_perm = np.random.permutation(len(lines))
    dev_size = int(len(lines) * dev_portion)
    dev_ids = rand_perm[:dev_size]
    new_test_ids = rand_perm[dev_size:]
    dev_lines = [lines[i] for i in dev_ids]
    new_test_lines = [lines[i] for i in new_test_ids]

    good_dev_file = os.path.join(scan_raw_folder, 'tasks_gooddev_mcd1.txt')
    new_test_file = os.path.join(scan_raw_folder, 'tasks_newtest_mcd1.txt')

    with open(good_dev_file, 'w') as fw:
        for line in dev_lines:
            fw.write(line)

    with open(new_test_file, 'w') as fw:
        for line in new_test_lines:
            fw.write(line)

def generate_scan_mcd2_good_dev(dev_portion=0.1):
    scan_raw_folder = os.path.join(ROOT_FOLDER, 'SCAN/mcd_split')
    gen_file = os.path.join(scan_raw_folder, 'tasks_test_mcd2.txt')

    with open(gen_file, 'r') as fr:
        lines = fr.readlines()
    rand_perm = np.random.permutation(len(lines))
    dev_size = int(len(lines) * dev_portion)
    dev_ids = rand_perm[:dev_size]
    new_test_ids = rand_perm[dev_size:]
    dev_lines = [lines[i] for i in dev_ids]
    new_test_lines = [lines[i] for i in new_test_ids]

    good_dev_file = os.path.join(scan_raw_folder, 'tasks_gooddev_mcd2.txt')
    new_test_file = os.path.join(scan_raw_folder, 'tasks_newtest_mcd2.txt')

    with open(good_dev_file, 'w') as fw:
        for line in dev_lines:
            fw.write(line)

    with open(new_test_file, 'w') as fw:
        for line in new_test_lines:
            fw.write(line)


def generate_scan_mcd3_good_dev(dev_portion=0.1):
    scan_raw_folder = os.path.join(ROOT_FOLDER, 'SCAN/mcd_split')
    gen_file = os.path.join(scan_raw_folder, 'tasks_test_mcd3.txt')

    with open(gen_file, 'r') as fr:
        lines = fr.readlines()
    rand_perm = np.random.permutation(len(lines))
    dev_size = int(len(lines) * dev_portion)
    dev_ids = rand_perm[:dev_size]
    new_test_ids = rand_perm[dev_size:]
    dev_lines = [lines[i] for i in dev_ids]
    new_test_lines = [lines[i] for i in new_test_ids]

    good_dev_file = os.path.join(scan_raw_folder, 'tasks_gooddev_mcd3.txt')
    new_test_file = os.path.join(scan_raw_folder, 'tasks_newtest_mcd3.txt')

    with open(good_dev_file, 'w') as fw:
        for line in dev_lines:
            fw.write(line)

    with open(new_test_file, 'w') as fw:
        for line in new_test_lines:
            fw.write(line)



parser = argparse.ArgumentParser()
parser.add_argument('--exp_type', type=str, help='experiment type')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--dev_portion', type=float, default=0.1, help='how much data from the original test are used as dev')


args = parser.parse_args()

np.random.seed(args.seed)

if args.exp_type == 'cogs':
    generate_cogs_good_dev(args.dev_portion)
elif args.exp_type == 'jump':
    generate_scan_addjump_good_dev(args.dev_portion)
elif args.exp_type == 'dax':
    generate_scan_adddax_good_dev(args.dev_portion)
elif args.exp_type == 'around_right':
    generate_scan_addaroundright_good_dev(args.dev_portion)
elif args.exp_type == 'mcd1':
    generate_scan_mcd1_good_dev(args.dev_portion)
elif args.exp_type == 'mcd2':
    generate_scan_mcd2_good_dev(args.dev_portion)
elif args.exp_type == 'mcd3':
    generate_scan_mcd3_good_dev(args.dev_portion)
else:
    raise ValueError

