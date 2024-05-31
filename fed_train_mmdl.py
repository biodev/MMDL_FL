#Adapted from NVFlare/examples/hello-world/step-by-step/cifar10/code/fl/train.py
#to support federated training of the https://github.com/hkmgeneis/MMDL model.

import argparse
import time
import copy
import os
from pathlib import Path  

#model file
from Netc import Cnn_With_Clinical_Net
from utils.custom_dset import CustomDset

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.nn import functional as F

from torchvision import datasets, models, transforms


import pandas as pd
import numpy as np
from sklearn import preprocessing

# (1) import nvflare client API
import nvflare.client as flare
from nvflare.app_common.app_constant import ModelName

from train_mmdl import train_model, test_model, preprocess_clinical

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--clinical_file", type=str, default="MMDL/label/clinic4_35.csv", nargs="?")
    parser.add_argument("--train_file", type=str, default="data/train_0.csv")
    parser.add_argument("--test_file", type=str, default="data/test_0.csv")
    parser.add_argument("--out_dir", type=str, default="output")
    parser.add_argument("--n_epochs", type=int, default="10")
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    trn_basename = os.path.basename(args.train_file)

    best_accuracy = 0.0

    clin_features, clin_pts = preprocess_clinical(args.clinical_file)
 
    net = Cnn_With_Clinical_Net(clin_features.shape[0]) 

    # (2) initialize NVFlare client API
    flare.init()

    # (3) run continously when launch_once=true
    while flare.is_running():

        # (4) receive FLModel from NVFlare
        input_model = flare.receive()
        client_id = flare.get_site_name()

        # Based on different "task" we will do different things
        # for "train" task (flare.is_train()) we use the received model to do training and/or evaluation
        # and send back updated model and/or evaluation metrics, if the "train_with_evaluation" is specified as True
        # in the config_fed_client we will need to do evaluation and include the evaluation metrics
        # for "evaluate" task (flare.is_evaluate()) we use the received model to do evaluation
        # and send back the evaluation metrics
        # for "submit_model" task (flare.is_submit_model()) we just need to send back the local model
        # (5) performing train task on received model
        if flare.is_train():
            print(f"({client_id}) current_round={input_model.current_round}, total_rounds={input_model.total_rounds}")

            # (5.1) loads model from NVFlare
            net.load_state_dict(input_model.params)

            #training code

            model_trn, train_acc, steps = train_model(net, args.train_file, clin_features, clin_pts, num_epochs=args.n_epochs)

            print(f"({client_id}) Finished Training")

            # (5.2) evaluation on local trained model to save best model

            local_accuracy, *_ = test_model(model_trn, args.test_file, clin_features, clin_pts)

            print(f"({client_id}) Evaluating local trained model: {local_accuracy}")
            if local_accuracy > best_accuracy:
                best_accuracy = local_accuracy
                torch.save(model_trn, args.out_dir+f'/model_fit_{trn_basename}.pt' )

            # (5.3) evaluate on received model for model selection

            net.load_state_dict(input_model.params)

            accuracy, *_ = test_model(net, args.test_file, clin_features, clin_pts)

            print(
                f"({client_id}) Evaluating received model for model selection: {accuracy}"
            )

            # (5.4) construct trained FL model
            output_model = flare.FLModel(
                params=model_trn.cpu().state_dict(),
                metrics={"accuracy": accuracy},
                meta={"NUM_STEPS_CURRENT_ROUND": steps},
            )

            # (5.5) send model back to NVFlare
            flare.send(output_model)
        
        # (6) performing evaluate task on received model
        elif flare.is_evaluate():
            net.load_state_dict(input_model.params)
            accuracy, *_ = test_model(net, args.test_file, clin_features, clin_pts)
            flare.send(flare.FLModel(metrics={"accuracy": accuracy}))

        # (7) performing submit_model task to obtain best local model
        elif flare.is_submit_model():
            model_name = input_model.meta["submit_model_name"]
            if model_name == ModelName.BEST_MODEL:
                try:
                    weights = torch.load(args.out_dir+f'/model_fit_{trn_basename}.pt')
                    net.load_state_dict(weights)
                    flare.send(flare.FLModel(params=net.cpu().state_dict()))
                except Exception as e:
                    raise ValueError("Unable to load best model") from e
            else:
                raise ValueError(f"Unknown model_type: {model_name}")