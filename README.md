# Overview

This project demonstrates the application of Federated Learning to improve news categorization models for different news agencies such as oxu.az, musavat.com, and axar.az. These agencies cannot share their proprietary data with others due to privacy concerns or competitive reasons. Federated Learning allows them to collaboratively improve their models without sharing their data.

# How Federated Learning Works

Federated Learning allows multiple parties to collaboratively train a machine learning model without sharing their data. The key steps are:

* Local Training: Each client (news agency) trains a local model on its own data.
* Weight Sharing: The local models' weights are sent to the server.
* Aggregation: The server aggregates the weights to update the global model.
* Model Update: The updated global model weights are sent back to the clients.
* Iteration: This process is repeated for a number of rounds until the model converges.

### Explanation

![Source](https://github.com/NijatZeynalov/Federated-Learning-for-News-Categorization/assets/31247506/6a1b5c37-6512-42bd-b85c-6a51a82fbff5)


The diagram above illustrates the Federated Learning system. This process ensures that the data never leaves the client's environment, preserving privacy and security while allowing collaborative learning.

> Central Server: This is where the global model resides.

> Global Model: The central server maintains the global model which aggregates updates from all clients.

> Clients: Each client represents a participating entity (e.g., a news agency) with its own local model.

> Local Model: Each client trains its local model on its own data.

> Model Updates: After local training, clients send their model updates (weights) to the central server.

> Global Model Update: The central server aggregates these updates to improve the global model and then sends the updated global model back to the clients.


# Benefits

* Privacy: Each news agency's data remains on-premises and is never shared.
* Collaboration: Agencies can collaboratively improve their models, benefiting from diverse datasets.
* Security: Only model updates (weights) are shared, reducing the risk of data breaches.

# Setup Instructions

Clone the repository:
```python
git clone https://github.com/nijatzeynalov/federated-learning-for-news-categorization.git
cd federated-learning-for-news-categorization
```
Install the required packages:

```python
pip install -r requirements.txt
```

# Data Source

The data used in this project is sourced from LocalDoc/news_azerbaijan_2. 

# Configure the parameters:

Edit the config.py file to set the desired parameters such as batch_size, epochs, and num_clients. Run the Federated Learning process:

```python
    python main.py
```

