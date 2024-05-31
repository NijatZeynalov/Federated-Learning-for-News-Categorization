import torch

batch_size = 16
epochs = 1
num_users = 3

args = {
    "batch_size": batch_size,
    "num_clients": num_users,
    "frac": 1,
    "ep_local": 1,
    "bs_local": batch_size,
    "epochs": epochs,
    "dataset": "text",
    "model": "BERT",
    "iid": "iid",

    # Unlearning params
    "unlearned_clients": [0],
    "t": 2,
    "r": 0.5
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")