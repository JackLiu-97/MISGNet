import torch
import random
import warnings
import torch.nn as nn
from utils.c_train_and_eval import evaluate
from utils.change_data import MyDataset

random.seed(47)
warnings.filterwarnings("ignore")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch fcn training")
    parser.add_argument("--ckpt_url",
                        default=r"D:\BaiduSyncdisk\建筑物论文2\权重文件\WHU\jack_2023083009t583828895\model_ema_best.pth",
                        help="data root"),
    parser.add_argument("--data_path", default=r"F:\Data_CD\WHU\test",
                        help="data root")
    parser.add_argument("--device", default="cuda", help="training device")
    args = parser.parse_args()
    return args


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = torch.load(args.ckpt_url, map_location=torch.device('cuda'))
    for m in model.modules():
        if isinstance(m, nn.Upsample):
            m.recompute_scale_factor = None
    model.eval()
    model.to(device)
    val_dataset = MyDataset(args.data_path)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=32,
                                             num_workers=0,
                                             pin_memory=False
                                             )
    confmat = evaluate(model, val_loader,
                       device=device,
                       num_classes=2, print_freq=500)
    val_info_print = str(confmat)
    print(val_info_print)


if __name__ == '__main__':
    args = parse_args()
    main(args)
