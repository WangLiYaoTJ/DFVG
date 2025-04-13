import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_filter", default="../pretrained_weights/filter.pth",type=str, help="the ckpt of neural filter network")
parser.add_argument("--fps", default=10, type=int, help="frame per second")
parser.add_argument('--gpu', type=int,     default=0, help='gpu device id')
parser.add_argument('--list_dir', type=str, default='./lists', help='path to list folder')
parser.add_argument('--dataset', type=str, default="ship", help='dataset to test')
parser.add_argument('--phase', type=str, default="test")
# process arguments
opts = parser.parse_args()
print(opts)
list_filename = os.path.join(opts.list_dir, "%s_%s.txt" % (opts.dataset, opts.phase))
with open(list_filename) as f:
    video_list = [line.rstrip() for line in f.readlines()]
for v in range(len(video_list)):
    video = video_list[v]
    depth_estimation_cmd = "python marigold/run.py --vid_name {}".format(video)
    os.system(depth_estimation_cmd)

    atlas_generation_cmd = "python atlas_and_filter/src/flow_guided_depth_atlas.py --vid_name {}".format(video)
    os.system(atlas_generation_cmd)

moveatlas_cmd = "python atlas_and_filter/moveatlas.py"
os.system(moveatlas_cmd)

filter_cmd = "python atlas_and_filter/src/neural_filter_net.py --fps {}".format(opts.fps)
os.system(filter_cmd)

deflicker_cmd = "python test.py"
os.system(deflicker_cmd)

fog_generation_cmd = "python networks/converthaze.py"
os.system(fog_generation_cmd)