"""Dropout RNN training utilities"""
import torch
from sklearn.metrics import accuracy_score, precision_score

class Metric:
    Y, Y_pred = None, None
    
    def collect(self, y: torch.Tensor, y_pred: torch.Tensor):
        assert len(y.shape) == 1 and y.shape == y_pred.shape
        if self.Y is None:
            self.Y, self.Y_pred = y, y_pred
        else:
            self.Y = torch.cat((self.Y, y))
            self.Y_pred = torch.cat((self.Y_pred, y_pred))
    
    def print_metric(self):
        raise Exception("Not implemented")


class BCMetric(Metric): 
    def accuracy(self):
        return accuracy_score(self.Y, torch.round(self.Y_pred))
    
    def print_metric(self):
        print("Accuracy = %.3f" % self.accuracy())


def train_model(model, trainloader, valloader, criterion, optimizer, path, epochs, patience=5, metrics={}):
    """Train model"""
    best_val_loss = float("inf")
    retry = 0
    for epoch in range(epochs):
        for phase in ["train", "validate"]:
            if phase == "train":
                model.train()
                batch_count = len(trainloader)
                dataloader = trainloader
            else:
                model.eval()
                batch_count = len(valloader)
                dataloader = valloader

            running_loss = 0.
            
            for i, (x, y) in enumerate(dataloader):
                if phase == "train":
                    optimizer.zero_grad()
                    output1, output2 = model(x.transpose(-2, -3))
                    loss = criterion(output1, output2, y) + model.regularizer(len(dataloader.dataset))
                    loss.backward()
                    optimizer.step()
                else:
                    with torch.no_grad():
                        output1, output2 = model(x.transpose(-2, -3))
                        loss = criterion(output1, output2, y) + model.regularizer(len(dataloader.dataset))
                        for metric in metrics:
                            metric.collect(y, output1)
                running_loss += loss.item()
                
                # print statistics
                print('[%d, %3.2f%%] %s loss: %.6f' %
                    (epoch + 1, (i+1)*100/batch_count, phase, loss.item()), end='\r')
            print('[%d, 100.00%%] %s loss: %.6f' %
                    (epoch + 1, phase, running_loss * dataloader.batch_size / len(dataloader.dataset)))
            
            if phase == "validate":
                for metric in metrics:
                    metric.print_metric()

                # Early stop
                if best_val_loss > running_loss: # improvement
                    best_val_loss = running_loss
                    retry = 0
                    torch.save(model.state_dict(), path)
                else:
                    retry += 1
                    print("Retry %d/%d" % (retry, patience))
                
                if retry >= patience:
                    model.load_state_dict(torch.load(path))
                    model.eval()
                    return model

    print("Finish training")
    model.load_state_dict(torch.load(path))
    model.eval()
    return model
