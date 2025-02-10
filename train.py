import time

import torch
from sklearn.metrics import f1_score
import argparse
from util import *
from model import *
import torch.nn.functional as F
def main(args):
    (
        shaGraph,
        hhGraph,
        ssGraph,
        kgOneHot,
        features,
        train_loader,
        dev_loader,
        test_loader,
        x_dev,
        x_test
    ) = load_data(args["x_h"],args["batch_size"])
    models = getAblationModels(args)
    index = 0
    for  describe,model in models.items():
        index = index+1
        if describe != "Attention3" :
            continue
        print(f"第{index}个模型,该模型的描述为：{describe}")
        modelName = "-"+str(index)+"-"+describe
        train(args, model, modelName, train_loader, shaGraph, hhGraph, ssGraph, kgOneHot, features, dev_loader, x_dev,
              test_loader, x_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("myHAN")
    parser.add_argument("-s", "--seed", type=int, default=1, help="Random seed")
    parser.add_argument(
        "-ld",
        "--log-dir",
        type=str,
        default="results",
        help="Dir for saving training results",
    )
    parser.add_argument(
        "--hetero",
        action="store_true",
        help="Use metapath coalescing with DGL's own dataset",
    )
    args = parser.parse_args().__dict__

    args = setup(args)

    main(args)
