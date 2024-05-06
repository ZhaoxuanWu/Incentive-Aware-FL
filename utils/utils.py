import os
import torch
import time
from torchtext.data import Batch


def train_model(model, loader, loss_fn, optimizer, device, local_epochs=1, **kwargs):
    model.train()
    total_loss = 0
    for _ in range(local_epochs):
        # running local epochs
        for _, batch in enumerate(loader):
            if isinstance(batch, Batch):
                # For NLP data
                data, label = batch.text, batch.label
                if isinstance(data, tuple):
                    data, label = (data[0].to(device), data[1].to(device)), label.to(device)
                else:
                    data, label = data.to(device), label.to(device)
            else:
                data, label = batch[0], batch[1]
                data, label = data.to(device), label.to(device)
            
            optimizer.zero_grad()
            pred = model(data)
            loss = loss_fn(pred, label)
            loss.backward()
            optimizer.step()
            total_loss += loss
    total_loss /= (local_epochs * len(loader))
        
    if 'scheduler' in kwargs: kwargs['scheduler'].step()

    return model, loss


def evaluate(model, eval_loader, device, loss_fn=None, verbose=False):
    model.eval()
    model = model.to(device)
    correct = 0
    total = 0
    loss = 0

    with torch.no_grad():
        for i, batch in enumerate(eval_loader):
            if isinstance(batch, Batch):
                # For the language datasets
                batch_data, batch_target = batch.text, batch.label
                if isinstance(batch_data, tuple):
                    batch_data, batch_target = (batch_data[0].to(device), batch_data[1].to(device)), batch_target.to(device)
                else:
                    batch_data, batch_target = batch_data.to(device), batch_target.to(device)
            else:
                # For the vision datasets
                batch_data, batch_target = batch[0], batch[1]
                batch_data, batch_target = batch_data.to(device), batch_target.to(device)

            outputs = model(batch_data)

            if loss_fn:
                loss += loss_fn(outputs, batch_target) * len(batch_target)
            else:
                loss = None

            correct += (torch.max(outputs, 1)[1].view(batch_target.size()).data == batch_target.data).sum()
            total += len(batch_target)

        accuracy =  correct.float() / total
        if loss_fn:
            loss /= total

    if verbose:
        print("Loss: {:.6f}. Accuracy: {:.4%}.".format(loss, accuracy))
    return loss, accuracy


def compute_grad_update(old_model, new_model, device=None):
    if device:
        old_model, new_model = old_model.to(device), new_model.to(device)
    
    global_dict = old_model.state_dict()
    
    for k in global_dict.keys():
        global_dict[k] = new_model.state_dict()[k].float() - old_model.state_dict()[k].float()
    return global_dict
        

def add_gradient_updates(grad_update_1, grad_update_2, weight = 1.0):
    assert len(grad_update_1) == len(
        grad_update_2), "Lengths of the two grad_updates not equal"
 
    if weight == 0:
        return

    for param_1, param_2 in zip(grad_update_1, grad_update_2):
        grad_update_1[param_1].data += grad_update_2[param_2].data * weight
  

def add_update_to_model(model, update, weight=1.0, device=None):
    if not update: return model
    # if device:
        # model = model.to(device)
        # update = [param.to(device) for param in update]
    
    global_dict = model.state_dict()
    
    for k in global_dict.keys():
        global_dict[k] = model.state_dict()[k].float() + weight * update[k].float()

    model.load_state_dict(global_dict)
    

def add_gradients_to_model(model, gradients, weights, device=None):
    
    global_dict = model.state_dict()
    
    for k in global_dict.keys():
        global_dict[k] = model.state_dict()[k].float()
        for i in range(len(gradients)):
            global_dict[k] += gradients[i][k] * weights[i]
  
    model.load_state_dict(global_dict)


def add_gradients_to_model_batch(models, gradients, weights, device=None):
    '''
        Slightly faster than looping add_gradients_to_model
        
        models: list of N models
        gradients: list of N gradients
        weights: list of N x N, where weights[i] stores the dim N weights for each gradient
    '''
    
    state_dicts = [model.state_dict() for model in models]
        
    for i, gradient in enumerate(gradients):
        for j, state in enumerate(state_dicts):
            for k in state_dicts[0].keys():
                state[k] = state[k].float()
                state[k] += gradient[k] * weights[j][i]

    for i, model in enumerate(models):
        model.load_state_dict(state_dicts[i])


def compute_grad_update_clip(old_model, new_model, device=None):
    if device:
        old_model, new_model = old_model.to(device), new_model.to(device)
    
    global_dict = old_model.state_dict()
    
    for k in global_dict.keys():
        global_dict[k] = torch.clip(new_model.state_dict()[k].float() - old_model.state_dict()[k].float(), min=-10, max=10)
    return global_dict


def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass