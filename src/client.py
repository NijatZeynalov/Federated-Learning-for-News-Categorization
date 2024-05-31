import torch
from transformers import AdamW
from src.utils import BaseModel

class Client(BaseModel):
    num_clients = 0

    def __init__(self, data, device):
        super().__init__()
        self.id = Client.num_clients
        self.dataloader = data
        self.__model = None
        self.model_record = []
        self.device = device
        self.unlearned = False
        Client.num_clients += 1

    def setup(self, args):
        self.local_epoch = args["ep_local"]

    @property
    def unlearned(self):
        return self.__unlearned

    @unlearned.setter
    def unlearned(self, unlearned):
        self.__unlearned = unlearned

    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, model):
        self.__model = model

    def client_update(self):
        self.model.train()
        self.model.to(self.device)

        optimizer = AdamW(self.model.parameters(), lr=2e-5)
        epoch_loss = []

        for epoch in range(self.local_epoch):
            batch_loss = []

            for batch in self.dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                self.model.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return self.model, sum(epoch_loss) / len(epoch_loss)

    def client_test(self):
        self.model.eval()
        self.model.to(self.device)
        test_loss, correct = 0, 0
        with torch.no_grad():
            for batch in self.dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                test_loss += outputs.loss.item()
                preds = outputs.logits.argmax(dim=1)
                correct += (preds == labels).sum().item()

        test_loss /= len(self.dataloader.dataset)
        test_accuracy = correct / len(self.dataloader.dataset)

        print(f"Average loss: {test_loss:.4f}, Accuracy: {100. * test_accuracy:.2f}%")

        return test_accuracy, test_loss