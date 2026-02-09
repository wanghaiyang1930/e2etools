"""
@brief: Train the dataset.
@author: wanghaiyang
@date: 2026-02-09
"""


import torch
import torch.nn as nn
import torch.optim as optim

def train(model, train_loader):
    epochs = 200
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        print(f'Epoch {epoch+1}/{epochs}:')
        for batch in train_loader:
            datas, labels = batch
            datas, labels = datas.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(datas)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(labels).sum().item()
            train_total += labels.size(0)

        print(f'Train Loss: {train_loss/len(train_loader):.4f}')
        print(f"Train Acc: {100.0*train_correct/train_total:.4f}%")
    

def evaluate(model, criterion, test_loader):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for batch in test_loader:
            datas, labels = batch
            datas, labels = datas.to(device), labels.to(device)

            outputs = model(datas)

            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

            print(f'Val Loss:{val_loss/len(test_loader):.4f}')
            print(f'Val Acc: {100.0*val_correct/val_total:.4f}%')

