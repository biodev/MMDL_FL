import argparse
from pathlib import Path
from train_mmdl import preprocess_clinical, test_model
import torch
import numpy as np

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--clinical_file", type=str, default="MMDL/label/clinic4_35.csv", nargs="?")
    parser.add_argument("--trained_model_file", type=str, default="data/test.pt")
    parser.add_argument("--test_file", type=str, default="data/test_0.csv")
    parser.add_argument("--out_dir", type=str, default="output")
    
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    clin_features, clin_pts = preprocess_clinical(args.clinical_file)

    print('Loading model...')

    model_trn = torch.load(args.trained_model_file)

    print('Starting testing...')

    test_acc, y_true, y_pred, preid_list, score_list = test_model (model_trn, args.test_file, clin_features, clin_pts)

    test_basename = Path(args.test_file).stem

    np.save(args.out_dir+f'/y_true_{test_basename}.npy', np.array(y_true)) 
    np.save(args.out_dir+f'/preid_{test_basename}.npy', np.array(preid_list))
    np.save(args.out_dir+f'/score_{test_basename}.npy', np.array(score_list))
    np.save(args.out_dir+f'/y_pred_{test_basename}.npy', np.array(y_pred))
    
    print('Testing complete...')
    print ('Accuracy of the network on test images: %d %%' % (
        100 * test_acc))