import os
import shutil
import os, sys, argparse,re

parser = argparse.ArgumentParser(description='Convert Haze')
parser.add_argument('--method', type=str, default="DFVG", help='test model name')
parser.add_argument('--dataset', type=str, default="ship", help='dataset to test')
parser.add_argument('--phase', type=str, default="test")
parser.add_argument('--data_dir', type=str, default='data', help='path to data folder')
parser.add_argument('--list_dir', type=str, default='lists', help='path to list folder')
args = parser.parse_args()
list_filename = os.path.join(args.list_dir, "%s_%s.txt" % (args.dataset, args.phase))
with open(list_filename) as f:
    video_list = [line.rstrip() for line in f.readlines()]
for A in range(len(video_list)):
    video = video_list[A]

    stage_all_folder = f'data/test/output/atlas/{video}/stage_all'
    if not os.path.exists(stage_all_folder):
        print(f"{stage_all_folder} dose not exist, skip")
        continue

    B_folders = [f for f in os.listdir(stage_all_folder) if os.path.isdir(os.path.join(stage_all_folder, f))]

    for B in B_folders:
        source_folder = os.path.join(stage_all_folder, B, 'stage_1', 'output')
        target_folder = f'data/test/output/atlas/{video}/stage_1/output/'

        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

        png_files = [f for f in os.listdir(source_folder) if f.endswith('.png')]

        # move
        for png in png_files:
            source_path = os.path.join(source_folder, png)
            target_path = os.path.join(target_folder, png)


            shutil.move(source_path, target_path)
            print(f" {png} has been moved from {source_path} to {target_path}")

for A in range(len(video_list)):
    v=A+1
    folder_path = f'data/test/output/atlas/{video}/stage_1/output/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    files = os.listdir(folder_path)

    def extract_number(file_name):

        match = re.match(r'(\d+)_(\d+)', file_name)
        if match:
            return int(match.group(1)), int(match.group(2))
        return (0, 0)


    # sort
    files_sorted = sorted(files, key=extract_number)

    # rename
    for index, file in enumerate(files_sorted, start=1):
        new_name = f"{index:06d}.png"
        file_path = os.path.join(folder_path, file)

        new_file_path = os.path.join(folder_path, new_name)

        os.rename(file_path, new_file_path)

        print(f"Renamed: {file} -> {new_name}")