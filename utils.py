# Dropout RNN training utilities

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
    return 0.5 * length_scale**2 * dropout_rate / precision / N


def bias_coefficient(length_scale, precision, N):
    """Calculate the coefficient of dropout rnn layer bias"""
    return 0.5 * length_scale**2 / precision / N


# TODO: early stop
def train_model(model, dataloader, criterion, optimizer, epochs):
    """Train model"""
    model.train()
    print_i = len(dataloader) // 5
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(x.transpose(1, 0))
            loss = criterion(outputs.flatten(), y.double())
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % print_i == print_i-1:
                print('[%d, %5d] loss: %.6f' %
                      (epoch + 1, i + 1, running_loss / print_i), end='\r')
                running_loss = 0.0               
        print()

    model.eval()
    print("Finish training")