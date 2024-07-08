import torch.nn as nn
import torch
import torch.optim.lr_scheduler as lr_scheduler
from typing import Any, Tuple, Callable
from sys import stdout

class In_between_epochs:
    def __call__(self, model:torch.nn.Module, loaders:dict[str,torch.utils.data.DataLoader], device:'torch.device|str', output_extraction_function:Callable, losses:dict) -> bool:
      raise NotImplementedError("Subclass must implement abstract method")

class NeuralNetwork(nn.Module):
  """
  This class implements a simple interface to get a working neural network using pytorch.
  """
  def __init__(self) -> None:
     super().__init__()
  def train_network(self,
            train_loader:torch.utils.data.DataLoader,
            validation_loader:torch.utils.data.DataLoader,
            optimizer:Any = torch.optim.Adam, 
            loss_function:'Any' = nn.CrossEntropyLoss(),
            learning_rate:float=.1,
            scheduler:'Callable[[Any], lr_scheduler.LRScheduler]|None' = None,
            epochs:int=10,
            batch_size:'int|None' = None,
            device:'torch.device|str'='cpu',
            output_extraction_function:Callable = lambda x: torch.round(x).detach().cpu(),
            metrics:dict[str,Callable] = {},
            in_between_epochs:dict[str,In_between_epochs] = {},
            verbose:bool=False,
            automatically_handle_gpu_memory:bool = True) -> Tuple[dict[str,list[float]],dict[str,list[float]]]:
    """
      A simple training loop for the neural network. It returns the epochs loss and accuracy history both on the training and the validation set. The tuple will be formatted as:
      train loss, train accuracy, val loss, val accuracy
      Parameters
      ----------
      train_loader: torch.utils.data.DataLoader
        A dataloader containing the dataset that will be used for training the network
      validation_loader: torch.utils.data.DataLoader
        A dataloader containing the dataset that will be used for validate the network at the end of each epoch
      optimizer:
        The optimizer to use while training, default to Adam.
      loss_function:
        The loss function to use while training, default to crossentropy
      learning_rate: float
        The learning rate that will be used in the optimizer to train the network. Default to .1
      epochs: int
        The number of training epochs, default to 10.
      device: str
        The device to use for the computation
      metrics: dict[str,callable]
        The metrics to use to evaluate the network
      verbose: bool
        Determines if intermidiate values of training statistics will be printed to stdout
      automatically_handle_gpu_memory: bool
        Determines if the training function should handle the moving of the data from e to the gpu memory (both the model and the training/validation data)
    """
    old_device = next(self.parameters()).device
    if next(self.parameters()).device == device or not automatically_handle_gpu_memory:
      net = self
    else:
      net = self.to(device)
    optimizer = optimizer(net.parameters(), learning_rate)
    lr_schedule = None
    if scheduler != None:
        lr_schedule = scheduler(optimizer)
    train_loss_history = []
    val_loss_history = []

    total_batch = int(len(train_loader.dataset) / train_loader.batch_size)
    train_metrics_scores = {}
    val_metrics_scores = {}
    for key in metrics:
        train_metrics_scores[key] = []
        val_metrics_scores[key] = []

    batch_size = train_loader.batch_size if batch_size is None else batch_size
    predicted_classes = []
    normal_labels = []
    for epoch in range(epochs):
        net.train()
        for batch_idx, data in enumerate(train_loader):
            labels = data[1]
            inputs = data[0]
            if automatically_handle_gpu_memory:
              inputs = self.__to(inputs, device)
              labels = self.__to(labels, device)
            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            predicted_classes += output_extraction_function(outputs)
            normal_labels += output_extraction_function(labels)
            batch = batch_idx * train_loader.batch_size
            if batch % batch_size == 0 or (batch_idx + 1) == total_batch:
              optimizer.zero_grad()
              if verbose:
                loss_str = "{:10.3f}".format(loss.cpu())
                str_metrics = {key: "{:10.3f}".format(metrics[key](normal_labels, predicted_classes)) for key in metrics.keys()}
                str_batch = str(batch_idx + 1)
                stdout.write(f"\rbatch {str_batch}/{total_batch} ----- loss: {loss_str} ----- {' ----- '.join([f'{key}: {str_metrics[key]}' for key in str_metrics.keys()])}")
                stdout.flush()
              predicted_classes = []
              normal_labels = []
            if automatically_handle_gpu_memory:
              self.__remove(inputs)
              torch.cuda.empty_cache()
        #
        # val_metrics, val_loss = self.__validate(validation_loader, metrics, loss_function, device, output_extraction_function, automatically_handle_gpu_memory)
        # for key in metrics:
        #   val_metrics_scores[key].append(val_metrics[key])
        # val_loss_history.append(val_loss)
        # train_metrics, train_loss = self.__validate(train_loader, metrics, loss_function, device, output_extraction_function, automatically_handle_gpu_memory)
        # for key in metrics:
        #   train_metrics_scores[key].append(train_metrics[key])
        # train_loss_history.append(train_loss)
        # if verbose:
        #   train_loss_str, val_loss_str = "{:10.3f}".format(train_loss_history[-1]), "{:10.3f}".format(val_loss_history[-1])
        #   train_metrics_score_str = {metric: "{:10.3f}".format(train_metrics_scores[metric][-1]) for metric in metrics.keys()}
        #   val_metrics_score_str = {metric: "{:10.3f}".format(val_metrics_scores[metric][-1]) for metric in metrics.keys()}
        #   out_str = f"EPOCH {epoch + 1} training loss: {train_loss_str} - validation loss: {val_loss_str}\n" + \
        #   '\n'.join([f"EPOCH {epoch + 1} training {metric}: {train_metrics_score_str[metric]} - validation {metric}: {val_metrics_score_str[metric]}" for metric in metrics.keys()]) +\
        #   f"\n{'-'*100}\n"
        #   stdout.write("\r" + " " * len(out_str) + "\r")
        #   stdout.flush()
        #   stdout.write(out_str)
        #   stdout.flush()
        #   print()
        if lr_schedule != None:
                lr_schedule.step()
        # loaders = {"train": train_loader, "validation": validation_loader}
        # losses = {"train": train_loss_history[-1], "validation": val_loss_history[-1]}
        # for in_between in in_between_epochs.keys():
            # result = in_between_epochs[in_between](self, loaders, device, output_extraction_function, losses)

            # if not type(result) == bool:
                # raise Exception(f"in between {in_between} returned a non-boolean result: {result}")
            # elif result:
                # if verbose:
                    # print(f"stopping after {epochs} epochs because of in between {in_between}")

                # train_metrics_scores['loss'] = train_loss_history
                # val_metrics_scores['loss'] = val_loss_history
                # return train_metrics_scores, val_metrics_scores


    # train_metrics_scores['loss'] = train_loss_history
    # val_metrics_scores['loss'] = val_loss_history

    if next(self.parameters()).device != old_device and automatically_handle_gpu_memory:
      del net
      self = self.to(old_device)
      torch.cuda.empty_cache()

    return train_metrics_scores, val_metrics_scores

  def __to(self, data, device):
    if isinstance(data, dict):
      return {key: self.__to(data[key], device) for key in data.keys()}
    elif isinstance(data, list):
      return [self.__to(d, device) for d in data]
    elif isinstance(data, tuple):
      return tuple([self.__to(d, device) for d in data])
    else:
      return data.to(device)
    
  def __remove(self, data):
    if isinstance(data, dict):
      for key in data.keys():
        self.__remove(data[key])
    elif isinstance(data, list) or isinstance(data, tuple):
      for d in data:
        self.__remove(d)
    else:
      del data

  def __validate(self, loader, metrics, loss_function, device, output_extraction_function, automatically_handle_gpu_memory):
    losses = []
    metrics_scores = {}
    for key in metrics.keys():
      metrics_scores[key] = []
    net = self.to(device)
    net.eval()
    with torch.no_grad():
        for _, data in enumerate(loader):
          labels = data[1]
          inputs = data[0]
          if automatically_handle_gpu_memory:
            inputs = self.__to(inputs, device)
            labels = self.__to(labels, device)
          outputs = net(inputs)
          loss = loss_function(outputs, labels)
          losses.append(loss)
          predicted_classes = output_extraction_function(outputs)
          real_labels = output_extraction_function(labels)
          for key in metrics.keys():
            metrics_scores[key].append(metrics[key](predicted_classes, real_labels))
          if automatically_handle_gpu_memory:
            self.__remove(inputs)
            torch.cuda.empty_cache()

    total_batch = int(len(loader.dataset) / loader.batch_size)
    average_loss = sum(losses)/(total_batch)
    mean_metrics_scores = {}
    for key in metrics.keys():
      mean_metrics_scores[key] = sum(metrics_scores[key])/len(loader)
    return mean_metrics_scores, average_loss

  def predict(self, loader:torch.utils.data.DataLoader, output_extraction_function:Callable, device:'str|torch.device|None' = None) -> list:
    net = self.to(device)
    net.eval()
    automatically_handle_gpu_memory = not device == None
    predictions = []
    with torch.no_grad():
        for _, data in enumerate(loader):
          labels = data[1]
          inputs = data[0]
          if automatically_handle_gpu_memory:
            inputs = self.__to(data[0], device)
          outputs = net(inputs)
          predictions += output_extraction_function(outputs)
          if automatically_handle_gpu_memory:
            self.__remove(inputs)
            del labels
            torch.cuda.empty_cache()

    return predictions
