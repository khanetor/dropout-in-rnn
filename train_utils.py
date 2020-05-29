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


class BCLogitMetric(Metric): 
    def accuracy(self):
        return accuracy_score(self.Y, torch.round(torch.sigmoid(self.Y_pred)))
    
    def print_metric(self):
        print("Accuracy = %.3f" % self.accuracy())


def train_model(model, trainloader, valloader, criterion, optimizer, path, epochs=100, patience=5, metrics={}):
    """Train model"""
    best_val_loss = float("inf")
    retry = 0
    for epoch in range(epochs):
        for phase in ["train", "validate"]:
            if phase == "train":
                model.train()
                dataloader = trainloader
            else:
                model.eval()
                dataloader = valloader

            batch_count = len(dataloader)
            running_loss = 0.
            
            for i, (x, y) in enumerate(dataloader):
                if phase == "train":
                    optimizer.zero_grad()
                    output = model(x.transpose(-2, -3))
                    loss = criterion(output, y)
                    loss_ = loss + model.regularizer() / len(dataloader.dataset)
                    loss_.backward()
                    optimizer.step()
                else:
                    with torch.no_grad():
                        output = model(x.transpose(-2, -3))
                        loss = criterion(output, y)
                        for metric in metrics:
                            metric.collect(y, output)
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
