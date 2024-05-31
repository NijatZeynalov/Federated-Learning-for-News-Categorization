# Overview

This project demonstrates the application of Federated Learning to improve news categorization models for different news agencies such as oxu.az, musavat.com, and axar.az. These agencies cannot share their proprietary data with others due to privacy concerns or competitive reasons. Federated Learning allows them to collaboratively improve their models without sharing their data.

# How Federated Learning Works

Federated Learning allows multiple parties to collaboratively train a machine learning model without sharing their data. The key steps are:

* Local Training: Each client (news agency) trains a local model on its own data.
* Weight Sharing: The local models' weights are sent to the server.
* Aggregation: The server aggregates the weights to update the global model.
* Model Update: The updated global model weights are sent back to the clients.
* Iteration: This process is repeated for a number of rounds until the model converges.

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

The data used in this project is sourced from LocalDoc/news_azerbaijan_2. This dataset contains news articles from various Azerbaijani news agencies and is used to train and evaluate the news categorization models.

# Configure the parameters:

Edit the config.py file to set the desired parameters such as batch_size, epochs, and num_clients. Run the Federated Learning process:

```python
    python main.py
```

