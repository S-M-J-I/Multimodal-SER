import torch
import gc
import random
from helpers.functions import *


def train_step(model: torch.nn.Module, dataloader, optimizer, loss_fn, idx2class, accuracy_fn=None, save_memory=False, device="cpu"):
    train_loss = 0.0
    train_acc = 0.0

    model.train()

    for batch, (videos, audios, labels, video_paths, audio_paths) in enumerate(dataloader):
        labels = labels.type(torch.LongTensor)
        videos, labels = videos.to(device), labels.to(device)

        y_logits, y_softmax = model(videos, audios)
        y_logits, y_softmax = y_logits.to(device), y_softmax.to(device)

        # print(y_logits.shape)

        optimizer.zero_grad()

        preds = y_softmax.argmax(dim=1).to(device)
        videos = videos.detach().cpu()
        # audios = audios.detach().cpu()
        del videos, audios
        torch.cuda.empty_cache()
        gc.collect()

        # print(labels.shape, preds.shape)

        loss = loss_fn(y_logits, labels)
        acc = accuracy_fn(preds, labels, num_classes=len(idx2class))
        train_loss += loss.item()
        train_acc += acc

        loss.backward()

        optimizer.step()

        if batch == 0 or batch == len(dataloader) - 1:
            sample = random.randint(1, y_logits.shape[0])-1
            print(
                f"Batch: #{batch} | Train Loss: {loss} | Train Accuracy: {acc}")
            show_example(video_paths[sample], audio_paths[sample], preds[sample].detach(
            ).cpu().item(), labels[sample].detach().cpu().item(), save_memory)

        del labels
        del video_paths
        del audio_paths
        preds = preds.detach().cpu()
        del preds
        y_logits = y_logits.detach().cpu()
        del y_logits
        torch.cuda.empty_cache()
        gc.collect()

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    print(
        f"Total Train loss: {train_loss} | Total Train accuracy: {train_acc}")
    return train_loss, train_acc


def eval_step(model: torch.nn.Module, dataloader, loss_fn, idx2class, accuracy_fn=None, save_memory=False, confusion_matrix=False, device="cpu"):
    eval_loss = 0.0
    eval_acc = 0.0

    y_true = []
    y_preds = []

    model.eval()

    with torch.no_grad():
        for batch, (videos, audios, labels, video_paths, audio_paths) in enumerate(dataloader):
            labels = labels.type(torch.LongTensor)
            videos, labels = videos.to(device), labels.to(device)

            y_logits, y_softmax = model(videos, audios)
            y_logits, y_softmax = y_logits.to(device), y_softmax.to(device)

            preds = y_softmax.argmax(dim=1).to(device)

            if confusion_matrix:
                y_preds.extend(preds.detach().cpu().numpy())
                y_true.extend(labels.detach().cpu().numpy())

            videos = videos.detach().cpu()
            # audios = audios.detach().cpu()
            del videos, audios
            torch.cuda.empty_cache()
            gc.collect()

            loss = loss_fn(y_logits, labels)
            acc = accuracy_fn(preds, labels, num_classes=len(idx2class))
            eval_loss += loss.item()
            eval_acc += acc

            if batch == 0 or batch == len(dataloader) - 1:
                sample = random.randint(1, y_logits.shape[0])-1
                print(
                    f"Batch: #{batch} | Eval. Loss: {loss} | Eval. Accuracy: {acc}")
                show_example(video_paths[sample], audio_paths[sample], preds[sample].detach(
                ).cpu().item(), labels[sample].detach().cpu().item(), save_memory)

            del labels
            del video_paths
            del audio_paths
            preds = preds.detach().cpu()
            del preds
            y_logits = y_logits.detach().cpu()
            del y_logits
            torch.cuda.empty_cache()
            gc.collect()

        eval_loss /= len(dataloader)
        eval_acc /= len(dataloader)

    print(f"Total Eval. Loss: {eval_loss} | Total Eval. Accuracy: {eval_acc}")

    if confusion_matrix:
        return eval_loss, eval_acc, y_true, y_preds
    else:
        return eval_loss, eval_acc
