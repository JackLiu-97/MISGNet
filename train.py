import os
import time
import torch
import datetime
import warnings
from ema_pytorch import EMA
from models.mynet.mynet4 import MyNet
from utils.change_data import MyDataset
from utils.c_distributed_utils import ConfusionMatrix
from utils.c_train_and_eval import train_one_epoch, evaluate, create_lr_scheduler

warnings.filterwarnings("ignore")

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    train_dataset = MyDataset(args.data_path)
    val_dataset = MyDataset(args.val_path)
    num_workers = 8

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=num_workers,
                                               shuffle=True,
                                               pin_memory=True
                                               )

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             num_workers=num_workers,
                                             pin_memory=True
                                             )
    model = MyNet(3, 2)
    model.to(device)
    model_ema = None
    if args.model_ema:
        model_ema = EMA(
            model,
            beta=0.9999,  # exponential moving average factor
            update_after_step=32,  # only after this number of .update() calls will it start updating
            update_every=1,  # how often to actually update, to save on compute (updates every 10th .update() call)
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs,
                                       warmup=True, warmup_epochs=1)
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        # 混合精度训练
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    # 开始时间
    start_time = time.time()
    best_F1 = 0.
    Last_epoch = 0
    save_path = os.path.join("output", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(save_path)
    for epoch in range(args.start_epoch, args.epochs):
        # print_model_info(model)
        mean_loss, lr = train_one_epoch(
            model, optimizer, train_loader, device, epoch,
            lr_scheduler=lr_scheduler,
            print_freq=args.print_freq,
            num_classes=args.num_classes + 1,
            scaler=scaler, ema=model_ema)
        if model_ema:
            confmat = evaluate(model_ema, val_loader,
                               device=device,
                               num_classes=args.num_classes + 1, print_freq=args.print_freq)
            val_info = ConfusionMatrix.todict(confmat)
            val_info_print = str(confmat)
            F1 = float(val_info['F1_Score'][1])
            print(val_info_print)

        if F1 == "nan":
            F1 = 0
        else:
            F1 = float(F1)
        save_txt = os.path.join(save_path, results_file)
        print(save_txt)
        with open(save_txt, "a") as f:
            # 记录每个epoch对应的train_loss、lr以及验证集各指标
            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {mean_loss:.4f}\n" \
                         f"lr: {lr:.6f}\n"

            f.write(train_info + val_info_print + "\n\n")

        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}
        if args.amp:
            save_file["scaler"] = scaler.state_dict()
        if F1 > best_F1:
            best_F1 = F1
            Last_epoch = epoch
            model_ema_name = str(epoch) + "model_ema_best.pth"
            model_name = str(epoch) + "model_best.pth"
            save_url_ema = os.path.join(save_path, model_ema_name)
            save_url = os.path.join(save_path, model_name)
            print(save_url)
            torch.save(model, save_url)
            torch.save(model_ema, save_url_ema)

        print("Best:", best_F1, )
        print("Best_epoch:", Last_epoch)
    print("best model in {} epoch".format(Last_epoch))
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch fcn training")
    parser.add_argument("--ckpt_url", default="", help="data root")
    parser.add_argument("--data_path", default=r"F:\Data_CD\LEVIR\256_2\train", help="data root")
    parser.add_argument("--val_path", default=r"F:\Data_CD\LEVIR\256_2\val", help="val root")
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("--out_path", default="output", help="val root")
    parser.add_argument("-b", "--batch-size", default=32, type=int)
    parser.add_argument("--epochs", default=150, type=int, metavar="N",
                        help="number of total epochs to train")
    parser.add_argument('--lr', default=0.0004, type=float, help='initial learning rate')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=100, type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")
    parser.add_argument("--model_ema", default=True, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")
    parser.add_argument("--seed", default=10, type=int,
                        help="Use torch.cuda.amp for mixed precision training")
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
