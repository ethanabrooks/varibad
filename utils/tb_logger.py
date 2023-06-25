import datetime
import json
import os
from glob import glob

import imageio
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)


class TBLogger:
    def __init__(self, args, exp_label, seed_list: list[int], debug=False):
        self.output_name = (
            exp_label
            + "_"
            + ",".join(map(str, seed_list))
            + "_"
            + datetime.datetime.now().strftime("_%d:%m_%H:%M:%S")
        )
        try:
            log_dir = args.results_log_dir
        except AttributeError:
            log_dir = args["results_log_dir"]
        if debug:
            log_dir = "/tmp"

        if log_dir is None:
            dir_path = os.path.abspath(
                os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)
            )
            dir_path = os.path.join(dir_path, "logs")
        else:
            dir_path = log_dir

        if not os.path.exists(dir_path):
            try:
                os.mkdir(dir_path)
            except:
                dir_path_head, dir_path_tail = os.path.split(dir_path)
                if len(dir_path_tail) == 0:
                    dir_path_head, dir_path_tail = os.path.split(dir_path_head)
                os.mkdir(dir_path_head)
                os.mkdir(dir_path)

        try:
            self.full_output_folder = os.path.join(
                os.path.join(dir_path, "logs_{}".format(args.env_name)),
                self.output_name,
            )
        except:
            self.full_output_folder = os.path.join(
                os.path.join(dir_path, "logs_{}".format(args["env_name"])),
                self.output_name,
            )

        self.writer = SummaryWriter(log_dir=self.full_output_folder)

        print("logging under", self.full_output_folder)

        if not os.path.exists(self.full_output_folder):
            os.makedirs(self.full_output_folder)
        with open(os.path.join(self.full_output_folder, "config.json"), "w") as f:
            try:
                config = {k: v for (k, v) in vars(args).items() if k != "device"}
            except:
                config = args
            config.update(device=self.device.type)
            json.dump(config, f, indent=2, cls=NumpyEncoder)

    def add(self, name, value, x_pos):
        self.writer.add_scalar(name, value, x_pos)

    def save_pngs(self, step: int):
        path_pattern = os.path.join(self.full_output_folder, "*.png")
        for path in glob(path_pattern):
            print("Saving png:", path)
            array = imageio.imread(path)
            tensor = torch.tensor(array)
            basename = os.path.basename(path)
            self.writer.add_image(basename, tensor, dataformats="HWC", global_step=step)
