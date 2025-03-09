
import torch


class DeepNetLinear(torch.nn.Module):
    """Deep Network with one hidden layer

    Args:
      n_inputs (int): number of input units
      n_hidden (int): number of units in hidden layer

    Attributes:
      in_layer (torch.nn.Linear): weights and biases of input layer
      out_layer (torch.nn.Linear): weights and biases of output layer

    """

    def __init__(self, n_inputs, n_hidden):
      super().__init__()  # needed to invoke the properties of the parent class torch.nn.Module
      self.in_layer = torch.nn.Linear(n_inputs, n_hidden) # neural activity --> hidden units
      self.out_layer = torch.nn.Linear(n_hidden, 1) # hidden units --> output

    def forward(self, r):
      """Decode stimulus orientation from neural responses

      Args:
        r (torch.Tensor): vector of neural responses to decode, must be of
          length n_inputs. Can also be a tensor of shape n_stimuli x n_inputs,
          containing n_stimuli vectors of neural responses

      Returns:
        torch.Tensor: network outputs for each input provided in r. If
          r is a vector, then y is a 1D tensor of length 1. If r is a 2D
          tensor then y is a 2D tensor of shape n_stimuli x 1.

      """
      h = self.in_layer(r)  # hidden representation
      y = self.out_layer(h)
      return y


class DeepNetReLU(torch.nn.Module):
    """ network with a single hidden layer h with a RELU """

    def __init__(self, n_inputs, n_hidden):
      super().__init__()  # needed to invoke the properties of the parent class nn.Module
      self.in_layer = torch.nn.Linear(n_inputs, n_hidden) # neural activity --> hidden units
      self.out_layer = torch.nn.Linear(n_hidden, 1) # hidden units --> output

    def forward(self, r):

      ############################################################################
      ## TO DO for students: write code for computing network output using a
      ## rectified linear activation function for the hidden units
      # Fill out function and remove
      # raise NotImplementedError("Student exercise: complete DeepNetReLU forward")
      ############################################################################

      h = torch.relu(self.in_layer(r)) # h is size (n_inputs, n_hidden)
      y = self.out_layer(h) # y is size (n_inputs, 1)


      return y


class CircularLoss():
    def __call__(self, input, target):
        return self.loss(input=input, target=target)


    def loss(self, input, target):
        """Calculates the mean absolute value of exp(input[i])-exp(target[i])

        Args:
            input (torch.Tensor): estimated angles (degrees)
            target (torch.Tensor): target angles (degrees)
        """
        input_radians = torch.deg2rad(input)
        target_radians = torch.deg2rad(target)
        answer = torch.mean(torch.abs(torch.exp(1j * input_radians) -
                                      torch.exp(1j * target_radians)))
        return answer


def train(net, optimizer, train_loss_fn, test_loss_fn, train_data, train_labels,
          test_data, test_labels, n_epochs=50, plot_every_n_iter=100):
    """Run gradient descent to optimize parameters of a given network

    Args:
      net (torch.nn.Module): PyTorch network whose parameters to optimize
      loss_fn: built-in PyTorch loss function to minimize
      train_data (torch.Tensor): n_train x n_neurons tensor with neural
        responses to train on
      train_labels (torch.Tensor): n_train x 1 tensor with orientations of the
        stimuli corresponding to each row of train_data
      n_epochs (int, optional): number of epochs of gradient descent to run
      learning_rate (float, optional): learning rate to use for gradient descent

    Returns:
      (list): training loss over iterations

    """

    # Placeholder to save the loss at each iteration
    train_loss = []
    test_loss = []

    # Loop over epochs
    for i in range(n_epochs):

        # compute network output from inputs in train_data
        out = net(train_data)  # compute network output from inputs in train_data

        # evaluate loss function
        train_loss_value = train_loss_fn(out, train_labels)

        # Clear previous gradients
        optimizer.zero_grad()

        # Compute gradients
        train_loss_value.backward()

        # Update weights
        optimizer.step()

        # Store current value of loss
        train_loss.append(train_loss_value.item())  # .item() needed to transform the tensor output of loss_fn to a scalar

        with torch.no_grad():
            out = net(test_data)
            test_loss_value = test_loss_fn(out, test_labels)
            test_loss.append(test_loss_value.item())

        # Track progress
        # if (i + 1) % (n_epochs // 5) == 0:
        if i % plot_every_n_iter == 0:
            print(f"iteration {i} | "
                  f"train loss: {train_loss_value.item():.3f} | "
                  f"test loss: {test_loss_value.item():.3f}")

    return train_loss, test_loss
