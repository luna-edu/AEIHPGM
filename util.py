import sys
import scipy.io
import time
from sklearn.model_selection import train_test_split
from model import *
from scipy.sparse import csc_matrix
import os
import pickle
import random
from pprint import pprint
import dgl
import numpy as np
import torch
from dgl.data.utils import _get_dgl_url, download, get_download_dir
from scipy import io as sio, sparse
import torch


def load_data(x_h, batch_size):
    buildAllGraph(x_h)
    data = sio.loadmat("./data/myData.mat")
    h_vs_s = data["HvsS"]
    h_vs_a = data["HvsA"]
    h_vs_h = data["HvsH"]
    s_vs_s = data["SvsS"]
    kgOneHot = torch.tensor(h_vs_a.toarray())
    shaGraph = dgl.heterograph(
        {
            ("herb", "hs", "symptom"): h_vs_s.nonzero(),

            ("symptom", "sh", "herb"): h_vs_s.transpose().nonzero(),
            ("herb", "ha", "attribute"): h_vs_a.nonzero(),
            ("attribute", "ah", "herb"): h_vs_a.transpose().nonzero(),
            ("herb", "hh", "herb"): h_vs_h.nonzero(),
            ("herb", "hh", "herb"): h_vs_h.transpose().nonzero(),
            ("symptom", "ss", "symptom"): s_vs_s.nonzero(),
            ("symptom", "ss", "symptom"): s_vs_s.transpose().nonzero(),
        }
    )
    hhGraph = dgl.graph(h_vs_h.nonzero(), num_nodes=811)
    ssGraph = dgl.graph((s_vs_s.nonzero()), num_nodes=390)
    features = torch.tensor([i for i in range(1236)])
    herb_list, symptom_list = getPrescriptions()
    pLen = len(herb_list)
    pS_list = [[0] * 390 for _ in range(pLen)]
    pS_array = np.array(pS_list)
    pH_list = [[0] * 811 for _ in range(pLen)]
    pH_array = np.array(pH_list)
    for i in range(pLen):
        pS_array[i, symptom_list[i]] = 1
        pH_array[i, herb_list[i]] = 1
    p_list = [x for x in range(pLen)]
    x_train, x_dev_test = train_test_split(p_list, test_size=0.4, shuffle=True,
                                           random_state=2024)
    x_dev, x_test = train_test_split(x_dev_test, test_size=1 - 0.5, shuffle=True, random_state=2024)
    print("train_size: ", len(x_train), "dev_size: ", len(x_dev), "test_size: ", len(x_test))
    train_dataset = presDataset(pS_array[x_train], pH_array[x_train])
    dev_dataset = presDataset(pS_array[x_dev], pH_array[x_dev])
    test_dataset = presDataset(pS_array[x_test], pH_array[x_test])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=batch_size, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=4)
    return (
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
    )

def valModel(args, criterion, shaGraph, hhGraph, ssGraph, kgOneHot, features, data_loader, data_index, model, isVal):
    epsilon = 1e-13
    checkpoint1 = time.time()
    dev_loss = 0

    dev_p5 = 0
    dev_p10 = 0
    dev_p20 = 0

    dev_r5 = 0
    dev_r10 = 0
    dev_r20 = 0

    for tsid, thid in data_loader:
        tsid, thid = tsid.float(), thid.float()
        # batch*805 概率矩阵
        if isVal == "dev":
            tsid = tsid.to(args["device"])
            thid = thid.to(args["device"])
            outputs = model(shaGraph, hhGraph, ssGraph, kgOneHot, features, tsid)
            dev_loss += criterion(outputs, thid).item()
            print('dev_loss: ', dev_loss / len(data_loader))
            return dev_loss
        model.to('cpu')
        shaGraph = shaGraph.to('cpu')
        hhGraph = hhGraph.to('cpu')
        ssGraph = ssGraph.to('cpu')
        kgOneHot = kgOneHot.to('cpu')
        features = features.to('cpu')
        tsid = tsid.to('cpu')
        thid = thid.to('cpu')
        outputs = model(shaGraph, hhGraph, ssGraph, kgOneHot, features, tsid)
        dev_loss += criterion(outputs, thid).item()
        for i, hid in enumerate(thid):
            trueLabel = []
            for idx, val in enumerate(hid):
                if val == 1:
                    trueLabel.append(idx)

            top5 = torch.topk(outputs[i], 5)[1]
            count = 0
            for m in top5:
                if m in trueLabel:
                    count += 1
            dev_p5 += count / 5
            dev_r5 += count / len(trueLabel)


            top10 = torch.topk(outputs[i], 10)[1]
            count = 0
            for m in top10:
                if m in trueLabel:
                    count += 1
            dev_p10 += count / 10
            dev_r10 += count / len(trueLabel)

            top20 = torch.topk(outputs[i], 20)[1]
            count = 0
            for m in top20:
                if m in trueLabel:
                    count += 1
            dev_p20 += count / 20
            dev_r20 += count / len(trueLabel)


    checkpoint2 = time.time()
    if isVal == "dev":
        print("验证过程总计执行耗时: {:.2f} 秒".format(checkpoint2 - checkpoint1))
        print('dev_loss: ', dev_loss / len(data_loader))
    else:
        print("测试过程总计执行耗时: {:.2f} 秒".format(checkpoint2 - checkpoint1))
        print('test_loss: ', dev_loss / len(data_loader))
    print('p5-10-20:', dev_p5 / len(data_index), dev_p10 / len(data_index), dev_p20 / len(data_index))
    print('r5-10-20:', dev_r5 / len(data_index), dev_r10 / len(data_index), dev_r20 / len(data_index))
    print('f1_5-10-20: ',
          2 * (dev_p5 / len(data_index)) * (dev_r5 / len(data_index)) / (
                  (dev_p5 / len(data_index)) + (dev_r5 / len(data_index)) + epsilon),
          2 * (dev_p10 / len(data_index)) * (dev_r10 / len(data_index)) / (
                  (dev_p10 / len(data_index)) + (dev_r10 / len(data_index)) + epsilon),
          2 * (dev_p20 / len(data_index)) * (dev_r20 / len(data_index)) / (
                  (dev_p20 / len(data_index)) + (dev_r20 / len(data_index)) + epsilon))
    return dev_loss

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def setup(args):
    default_configure = {
        "lr": 3e-4,  # Learning rate
        "num_heads": [6],  # Number of attention heads for node-level attention
        "dropout": 0.2,
        "weight_decay": 3e-3,  # 3e-4
        "num_epochs": 200,
        "patience": 7,
        "hidden_dim": 128,
        "x_h": 80,
        "batch_size": 512
    }
    args.update(default_configure)
    set_random_seed(args["seed"])
    args["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"
    timeStr = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
    logPath = os.path.join("trainLog", timeStr + ".log")
    sys.stdout = Logger(logPath)
    print(timeStr)
    ptPath = os.path.join("trainPt", timeStr)
    os.makedirs(ptPath)
    ptPath = os.path.join(ptPath, "checkpoint.pt")
    args["ptPath"] = ptPath
    args["logPath"] = logPath
    return args

def set_random_seed(seed=0):
    """Set random seed.
    Parameters
    ----------
    seed : int
        Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # 这里可以添加对self.terminal和self.log的flush调用，以确保所有内容都被正确输出
        self.terminal.flush()
        self.log.flush()

    def close(self):
        # 当不再需要日志文件时，应确保它被正确关闭
        self.log.close()

def buildAllGraph(x_h):

    hhGraph = getHHorSSGraph(txtPath='./data/TCM/pre_herbs.txt', T=x_h, num_nodes=811)
    ssGraph = getHHorSSGraph(txtPath='./data/TCM/pre_symptoms.txt', T=5, num_nodes=390)

    h2sGraph = getCSCH2S()
    h2aGraph = getCSCH2A(kgHerbAttributePath='./data/kgHerbAttributeAndMyAttribute.txt')
    data = {"HvsS": h2sGraph, "HvsA": h2aGraph, "HvsH": hhGraph, "SvsS": ssGraph}
    scipy.io.savemat('data/myData.mat', data)

def getHHorSSGraph(txtPath, T, num_nodes):
    with open(txtPath, 'r', encoding='utf-8') as h:
        freq = {}
        for temp in h.readlines():
            herbList = set()
            for item in temp.strip().split(" "):
                herbList.add(int(item))
            n = len(herbList)
            herbList = list(herbList)
            herbList.sort()
            for i in range(n):
                for j in range(i + 1, n):
                    edge = str(herbList[i]) + "-" + str(herbList[j])
                    if edge in freq:
                        freq[edge] = freq[edge] + 1
                    else:
                        freq[edge] = 1
    res = []
    for k in freq:
        if freq[k] >= T:
            res.append([k.split("-")[0], k.split("-")[1]])
            res.append([k.split("-")[1], k.split("-")[0]])
    adj_matrix = np.zeros((num_nodes, num_nodes))
    for edge in res:
        node1 = int(edge[0])
        node2 = int(edge[1])
        adj_matrix[node1, node2] = 1
        adj_matrix[node2, node1] = 1
    adj_csc = csc_matrix(adj_matrix)
    return adj_csc

def getCSCH2S():
    herbs_list, symptoms_list = getPrescriptions()
    herbIndices = []
    symptomIndices = []
    for i in range(len(herbs_list)):
        herbs = herbs_list[i]
        symptoms = symptoms_list[i]
        for herb in herbs:
            for symptom in symptoms:
                herbIndices.append(herb)
                symptomIndices.append(symptom)
    herbIndices = np.array(herbIndices)
    symptomIndices = np.array(symptomIndices)
    data = np.ones(len(herbIndices))
    csc = csc_matrix((data, (herbIndices, symptomIndices)))
    csc.data = np.ones_like(csc.data)
    return csc

def getCSCH2A(kgHerbAttributePath):
    herbIndices = []
    attributeIndices = []
    with open(kgHerbAttributePath, 'r', encoding='utf-8') as x:
        for herbIndex, temp in enumerate(x):
            attributes = [1 if item in ["0.5", "2"] else int(item) for item in temp.strip().split(" ")]
            attributesIndices = [i for i, y in enumerate(attributes) if y == 1]
            herbIndices.extend([herbIndex] * len(attributesIndices))
            attributeIndices.extend(attributesIndices)
    herbIndices = np.array(herbIndices)
    symptomIndices = np.array(attributeIndices)
    data = np.ones(len(herbIndices))
    csc = csc_matrix((data, (herbIndices, symptomIndices)))
    csc.data = np.ones_like(csc.data)
    return csc

def getPrescriptions():
    herbs_list = []
    with open("./data/TCM/pre_herbs.txt", 'r', encoding='utf-8') as herbs_text:
        for temp in herbs_text.readlines():
            herbs_list.append([int(item) for item in temp.strip().split(" ")])
    symptoms_list = []
    with open("./data/TCM/pre_symptoms.txt", 'r', encoding='utf-8') as symptoms_text:
        for temp in symptoms_text.readlines():
            symptoms_list.append([int(item) for item in temp.strip().split(" ")])
    return herbs_list, symptoms_list


def train(args, model, modelName, train_loader, shaGraph, hhGraph, ssGraph, kgOneHot, features, dev_loader, x_dev,
          test_loader, x_test):
    stopper = EarlyStopping(patience=args["patience"], verbose=True, path=args["ptPath"] + modelName)
    criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.8)
    # lossList = []
    start_time = time.time()
    model.to(args["device"])
    for epoch in range(args["num_epochs"]):
        model.train()
        running_loss = 0.0
        for i, (sid, hid) in enumerate(train_loader):
            # sid torch.Size([512, 390]), hid torch.Size([512, 805])
            sid, hid = sid.float(), hid.float()
            optimizer.zero_grad()
            shaGraph = shaGraph.to(args["device"])
            hhGraph = hhGraph.to(args["device"])
            ssGraph = ssGraph.to(args["device"])
            kgOneHot = kgOneHot.to(args["device"])
            features = features.to(args["device"])
            sid = sid.to(args["device"])
            hid = hid.to(args["device"])
            outputs = model(shaGraph, hhGraph, ssGraph, kgOneHot, features, sid)
            loss = criterion(outputs, hid)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print('[Epoch {}]train_loss: '.format(epoch + 1), running_loss / len(train_loader))
        checkpoint1 = time.time()
        print("第{}次训练执行耗时: {:.2f} 秒".format(epoch + 1, checkpoint1 - start_time))
        scheduler.step()
        if epoch >= args["num_epochs"] - 10:
            model.eval()
            dev_loss = valModel(args, criterion, shaGraph, hhGraph, ssGraph, kgOneHot, features, dev_loader, x_dev,
                                model, "dev")
            stopper(dev_loss / len(dev_loader), model)
            if stopper.early_stop:
                print("Early stopping")
                break
    valModel(args, criterion, shaGraph, hhGraph, ssGraph, kgOneHot, features, test_loader, x_test, model, "test")

def getAblationModels(args):
    modelAtt3 = MyHANAtt3(
        meta_paths_herb=[["ha", "ah"], ["hs", "sh"], ["hh"]],
        meta_paths_symptom=[["sh", "hs"], ["sh", "ha", "ah", "hs"], ["ss"]],
        meta_paths_attribute=[["ah", "ha"], ["ah", "hs", "sh", "ha"]],  # 这个试了试没啥用就去掉了
        in_size=64,
        hidden_size=args["hidden_dim"],
        out_size=811,
        num_heads=args["num_heads"],
        dropout=args["dropout"],
    )

    return {"Attention3": modelAtt3}

class presDataset(torch.utils.data.Dataset):
    def __init__(self, a, b):
        self.pS_array, self.pH_array = a, b

    def __getitem__(self, idx):
        sid = self.pS_array[idx]
        hid = self.pH_array[idx]
        return sid, hid

    def __len__(self):
        return self.pH_array.shape[0]

