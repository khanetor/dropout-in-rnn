"""Dropout RNN training utilities"""
import torch

def filter_parameters(param_gen, prefix, postfix):
    """Filter parameters by names
    -Args:
       param_gen: named_parameters
       prefix: prefix of parameter names
       postfix: postfix of parameter names"""

    filtered = filter(lambda named_params: named_params[0].startswith(prefix), param_gen)
    filtered = filter(lambda named_params: named_params[0].endswith(postfix), filtered)
    return map(lambda named_params: named_params[1], filtered)


def weight_coefficient(length_scale, precision, dropout_rate, N):
    """Calculate the coefficient of dropout rnn layer weight"""
    return 0.5 * length_scale**2 * (1-dropout_rate) / precision / N


def bias_coefficient(length_scale, precision, N):
    """Calculate the coefficient of dropout rnn layer bias"""
    return 0.5 * length_scale**2 / precision / N


# TODO: early stop
def train_model(model, dataloader, criterion, optimizer, epochs):
    """Train model"""
    model.train()
    batch_count = len(dataloader)
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()
            output1, output2 = model(x.transpose(-2, -3))
            loss = criterion(output1, output2, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # print statistics
            print('[%d, %3.2f%%] loss: %.6f' %
                (epoch + 1, (i+1)*100/batch_count, loss.item()), end='\r')
        print('[%d, 100.00%%] loss: %.6f' %
                (epoch + 1, running_loss / batch_count))

    model.eval()
    print("Finish training")