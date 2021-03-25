import torch
import config
import logging
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import ticker
import torch.nn.functional as F
from model import Code2Vec, Code2VecEncoder
from dataset import Code2VecDataset, load_vocabularies
logging.basicConfig(level = logging.INFO)


class DRCLoss(torch.nn.Module):  
    def __init__(self):
        super(DRCLoss, self).__init__()

    def forward(self, ap1, af1, ap2, af2):
        batch_size = ap1.shape[0]
        # ap1 - [batch_size, target dim]
        # af1 - [batch_size, target_dim]

        # Compute assignment feature loss
        af_loss = - 1 / batch_size * torch.sum( torch.log( F.softmax(torch.div(torch.mm(torch.transpose(af1, 0, 1), af2), config.TEMPERATURE ), dim=1) ) )

        # Compute assignment probability loss
        ap_loss = - 1 / batch_size * torch.sum( torch.log( F.softmax(torch.div(torch.mm(torch.transpose(ap1, 0, 1), ap2), config.TEMPERATURE ), dim=1) ) )

        # Compute regularization loss
        regularization_loss = 1 / batch_size * torch.sum( torch.square( torch.sum(ap1, dim=0)))

        return af_loss + ap_loss + config.LAMBDA * regularization_loss


def train(epochs, lr=0.001):
    logging.info(f'Training model on {config.DEVICE}')
    word_vocab, path_vocab, label_vocab = load_vocabularies(f'./data/{config.DATASET}/{config.DATASET}.dict.c2v')
    train_ds = Code2VecDataset(f'./data/{config.DATASET}/{config.DATASET}.train.c2v', word_vocab, path_vocab, label_vocab)
    train_aug_ds = Code2VecDataset(f'./data/{config.DATASET}/{config.DATASET}.train_aug.c2v', word_vocab, path_vocab, label_vocab)
    
    val_ds = Code2VecDataset(f'./data/{config.DATASET}/{config.DATASET}.val.c2v', word_vocab, path_vocab, label_vocab)
    val_aug_ds = Code2VecDataset(f'./data/{config.DATASET}/{config.DATASET}.val_aug.c2v', word_vocab, path_vocab, label_vocab)
    
    train_dataloader = torch.utils.data.DataLoader(train_ds, batch_size = config.BATCH_SIZE, num_workers = 5)
    train_aug_dataloader = torch.utils.data.DataLoader(train_aug_ds, batch_size = config.BATCH_SIZE, num_workers = 5)

    val_dataloader = torch.utils.data.DataLoader(val_ds, batch_size = config.BATCH_SIZE, num_workers = 5)
    val_aug_dataloader = torch.utils.data.DataLoader(val_aug_ds, batch_size = config.BATCH_SIZE, num_workers = 5)


    logging.info(f'Loaded {len(train_dataloader.dataset)} samples for training and {len(val_dataloader.dataset)} samples for validation')
    encoder = Code2VecEncoder(len(word_vocab), len(path_vocab), config.EMBEDDING_DIM, config.CODE_VECTOR_DIM, config.DROPOUT)
    model = Code2Vec(encoder, config.TARGET_DIM)
    model = model.to(config.DEVICE)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = DRCLoss().to(config.DEVICE)

    history = {'train_loss': list(), 'eval_loss': list()}
    def after_epoch(epoch):
        evaluate(model, history, loss_fn, val_dataloader, val_aug_dataloader, epoch, label_vocab)
        torch.save(model.encoder.state_dict(), f'./models/encoder.ckpt')
        torch.save(model.state_dict(), f'./models/code2vec.ckpt')
        logging.info(f'[epoch {epoch}] model saved')

    for epoch in range(1, epochs + 1):
            train_epoch(model, optimizer, loss_fn, history, train_dataloader, train_aug_dataloader, epoch, label_vocab, after_epoch_callback=after_epoch)
    save_history(history)


def train_epoch(model, optimizer, loss_fn, history, dataloader, dataloader_aug, epoch_idx, label_vocab, after_epoch_callback=None):
    total_loss, batch_cnt, data_cnt = 0, 0, 0
    for batch_nr, (x1, x2) in enumerate(zip(dataloader, dataloader_aug), 1):
        
        label1, x_s1, path1, x_t1 = x1
        label1, x_s1, path1, x_t1 = label1.to(config.DEVICE), x_s1.to(config.DEVICE), path1.to(config.DEVICE), x_t1.to(config.DEVICE)

        label2, x_s2, path2, x_t2 = x2
        label2, x_s2, path2, x_t2 = label2.to(config.DEVICE), x_s2.to(config.DEVICE), path2.to(config.DEVICE), x_t2.to(config.DEVICE)

        model.train()
        optimizer.zero_grad()

        # Get assignment probability and features for the sample and the augmented sample
        ap1, af1, ap2, af2 = model((x_s1, path1, x_t1), (x_s2, path2, x_t2))
        
        loss = loss_fn(ap1, af1, ap2, af2)

        data_cnt += label1.shape[0]
        total_loss += loss.item() * label1.shape[0]

        loss.backward()
        optimizer.step()
    history['train_loss'].append(total_loss / data_cnt)
    if after_epoch_callback is not None:
        after_epoch_callback(epoch_idx)


def evaluate(model, history, loss_fn, dataloader, dataloader_aug, epoch_idx, label_vocab):
    model.eval()

    total_loss, batch_cnt, data_cnt = 0, 0, 0
    with torch.no_grad():
        for i, (x1, x2) in enumerate(zip(dataloader, dataloader_aug), 1):
            
            label1, x_s1, path1, x_t1 = x1
            label1, x_s1, path1, x_t1 = label1.to(config.DEVICE), x_s1.to(config.DEVICE), path1.to(config.DEVICE), x_t1.to(config.DEVICE)

            label2, x_s2, path2, x_t2 = x2
            label2, x_s2, path2, x_t2 = label2.to(config.DEVICE), x_s2.to(config.DEVICE), path2.to(config.DEVICE), x_t2.to(config.DEVICE)

            ap1, af1, ap2, af2 = model((x_s1, path1, x_t1), (x_s2, path2, x_t2))
            loss = loss_fn(ap1, af1, ap2, af2)

            data_cnt += label1.shape[0]
            total_loss += loss.item() * label1.shape[0]
            batch_cnt += 1

    history['eval_loss'].append(total_loss / data_cnt)
    logging.info(f'[epoch {epoch_idx} eval] loss: {total_loss / data_cnt}')



def save_history(history):
    for metric, values in history.items():
        # save raw data
        with open(f'./logs/{metric}.data', mode='w') as f:
            data = ','.join([str(v) for v in values])
            f.write(data)
    # save graph
    train_loss = history['train_loss']
    eval_loss = history['eval_loss']
    x = np.linspace(1, len(train_loss), len(train_loss))
    plt.figure()
    plt.plot(x, train_loss, marker='o', label='train_loss')
    plt.plot(x, eval_loss, marker='*', label='eval_loss')
    plt.title("Loss during training")
    plt.xlabel('Epoch')
    plt.ylabel("Loss")
    plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
    plt.legend()
    plt.savefig(f'./logs/loss.png')
    plt.close()


if __name__ == '__main__':
    train(config.EPOCHS)