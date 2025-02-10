import argparse
from util import *
from model import MyHANAtt3

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

    model = MyHANAtt3(
        meta_paths_herb=[["ha", "ah"], ["hs", "sh"], ["hh"]],
        meta_paths_symptom=[["sh", "hs"], ["sh", "ha", "ah", "hs"], ["ss"]],
        meta_paths_attribute=[["ah", "ha"], ["ah", "hs", "sh", "ha"]],  # 这个试了试没啥用就去掉了
        in_size=64,
        hidden_size=args["hidden_dim"],
        out_size=811,
        num_heads=args["num_heads"],
        dropout=args["dropout"],
    )
    model.load_state_dict(torch.load("xxx"))
    model.eval()
    criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")  # 多标签
    valModel(args,criterion, shaGraph,hhGraph,ssGraph,kgOneHot, features, test_loader, x_test, model ,"test")

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
