import copy
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import BertForSequenceClassification
from src.utils import BaseModel

class Server(BaseModel):
    def __init__(self, num_labels, device):
        super().__init__()
        self.round = 0
        self.model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-uncased', num_labels=num_labels).to(device)
        self.w_glob = None
        self.device = device

        self.history = {
            "train_loss": [],
            "test_loss": [],
            "train_acc": [],
            "test_acc": []
        }

    def setup(self, args, test_loader):
        self.num_clients = args["num_clients"]
        self.num_rounds = args["epochs"]
        self.local_epochs = args["ep_local"]
        self.batch_size = args["batch_size"]
        self.clients = args["clients"]
        self.unlearned_clients = args["unlearned_clients"]
        self.test_loader = test_loader

        self.local_model_record = [[] for _ in range(self.num_clients)]
        self.global_model_record = []

    def fedAvg(self, w):
        w_avg = copy.deepcopy(w[0])
        for k in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[k] += w[i][k]
            w_avg[k] = torch.div(w_avg[k], len(w))
        return w_avg

    def send_global_model(self):
        for client in self.clients:
            client.model = copy.deepcopy(self.model)
            client.model_record.append(client.model)

    def train_global_model(self):
        w_locals = [self.w_glob for _ in range(self.num_clients)]
        loss_locals = [0 for _ in range(self.num_clients)]

        self.send_global_model()
        print("Send global model to all clients...")

        acc_test_clients, loss_test_clients = [], []
        for id, client in enumerate(self.clients):
            print(f"\nUpdating client {id}...")
            local_model, loss_train_client = client.client_update()

            self.local_model_record[id].append(local_model)
            w_locals.append(local_model.state_dict())
            loss_locals.append(loss_train_client)

            print(f"\nEvaluating client {id}...")
            acc_test_client, loss_test_client = client.client_test()
            acc_test_clients.append(acc_test_client)
            loss_test_clients.append(loss_test_client)

        w_glob = self.fedAvg(w_locals)
        self.model.load_state_dict(w_glob)

        return acc_test_clients, loss_locals

    def federated_learning(self):
        self.model.train()
        self.w_glob = self.model.state_dict()

        print("\tFederated Learning:")
        for round in tqdm(range(self.num_rounds)):
            self.round = round + 1
            print(f"\nRound {self.round}/{self.num_rounds}: Starting...")
            acc_train_clients, loss_train_clients = self.train_global_model()
            self.history["train_acc"].append(100 * sum(acc_train_clients) / len(acc_train_clients))
            self.history["train_loss"].append(sum(loss_train_clients) / len(loss_train_clients))

            print(f"\nRound {self.round}: Evaluating...")
            acc_test_server, loss_test_server = self.test_global_model()
            self.history["test_acc"].append(acc_test_server)
            self.history["test_loss"].append(loss_test_server)

            print(f"|---- Average Clients Loss: {sum(loss_train_clients) / len(loss_train_clients)}")
            print(f"|---- Average Clients Accuracy: {100 * sum(acc_train_clients) / len(acc_train_clients):.2f}%")
            print(f"|---- Server Testing Accuracy: {acc_test_server:.2f}%")

            print(f"\nRound {self.round}: Finished!\n")
            print(f"---------------------------------")

        self.show_result()
        self.plot(self.history)

    def show_result(self):
        acc_test_server, loss_test_server = self.test_global_model()
        print(f' \n Results after {self.num_rounds} global rounds of training:')
        print(f"|---- Testing Accuracy: {acc_test_server:.2f}%")

        print(f"\nUnlearned Clients:")
        for i in self.unlearned_clients:
            acc_test, loss_test = self.clients[i].client_test()
            print(f"|---- Unlearned Client - {i} Accuracy: {100 * acc_test:.2f}%")

    def plot(self, history):
        fig, axs = plt.subplots(2, 1)
        axs[0].plot(history["train_loss"], color="b", label="Training Loss")
        axs[0].plot(history["test_loss"], color='r', label="Testing Loss")
        legend = axs[0].legend(loc="best", shadow=True)
        axs[0].set_xlabel("Communication Rounds")
        axs[0].set_ylabel("Loss")

        axs[1].plot(history["train_acc"], color="b", label="Training Accuracy")
        axs[1].plot(history["test_acc"], color='r', label="Testing Accuracy")
        legend = axs[1].legend(loc="best", shadow=True)
        axs[1].set_xlabel("Communication Rounds")
        axs[1].set_ylabel("Accuracy")

    def test_global_model(self):
        self.model.eval()

        test_loss = 0
        correct = 0
        with torch.no_grad():
            for batch in self.test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                test_loss += outputs.loss.item()
                preds = outputs.logits.argmax(dim=1)
                correct += (preds == labels).sum().item()

        test_loss /= len(self.test_loader.dataset)
        accuracy = 100. * correct / len(self.test_loader.dataset)

        print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(self.test_loader.dataset)} ({accuracy:.2f}%)\n')

        return accuracy, test_loss