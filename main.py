import pandas as pd
from torch.utils.data import DataLoader
from src.client import Client
from src.server import Server
from src.utils import TextDataset, convert_labels
import src.config as config

client_datasets = []
all_labels = []
for i in range(config.args["num_clients"]):
    csv_path = f'./data/client_{i}.csv'
    df = pd.read_csv(csv_path)
    texts = df['text'].tolist()
    labels = df['category'].tolist()
    labels, label_dict = convert_labels(labels)
    all_labels.extend(labels)
    client_datasets.append(TextDataset(texts, labels))

num_labels = len(set(all_labels))

local_datasets = []
for i in range(config.args["num_clients"]):
    local_datasets.append(DataLoader(client_datasets[i], batch_size=config.args["bs_local"], shuffle=True))

test_csv_path = './data/client_0.csv'
test_df = pd.read_csv(test_csv_path)
test_texts = test_df['text'].tolist()
test_labels = test_df['category'].tolist()
test_labels, _ = convert_labels(test_labels)
test_dataset = TextDataset(test_texts, test_labels)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

config.args["clients"] = []
for i in range(config.args["num_clients"]):
    client = Client(local_datasets[i], config.device)
    client.setup(config.args)
    config.args["clients"].append(client)

for unlearned in config.args["unlearned_clients"]:
    config.args["clients"][unlearned].unlearned = True

server = Server(num_labels, config.device)
server.setup(config.args, test_loader)

server.federated_learning()
