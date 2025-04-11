import torch


def validate(model, val_dataloader, criterion, device):
    logs = {
        'loss': 0,
    }

    model.eval()
    

    with torch.no_grad():
        for image_features, input_seqs, target_seqs in val_dataloader:
            image_features = image_features.to(device)
            input_seqs = input_seqs.to(device)
            target_seqs = target_seqs.to(device)

            outputs = model(image_features, input_seqs)
            outputs = outputs.reshape(-1, outputs.size(-1)) 
            target_tensor = target_seqs.view(-1)  
       
            loss = criterion(outputs, target_tensor.long())
            logs['loss'] += loss.item()


    logs['loss'] /= len(val_dataloader)
   

    return logs
