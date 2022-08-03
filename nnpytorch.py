import torch 
from torch import nn, optim
# melakukan mco
model = nn.Sequential(
    nn.Linear(7, 16),
    nn.ReLU(),
    nn.Linear(16, 8),
    nn.ReLU(),
    nn.Linear(8, 4),
    nn.ReLU(),
    nn.Linear(4, 2),
    nn.CrossEntropyLoss(1)
)  
criterion = nn.BCELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)
