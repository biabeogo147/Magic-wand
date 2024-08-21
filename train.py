import os
import torch
import dataset
import argparse
import numpy as np
import magic_wand_model
from pprint import pprint
from tqdm.autonotebook import tqdm
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score


def get_args():
    parser = argparse.ArgumentParser(description='magic wand training script')
    parser.add_argument('--num_workers', '-n', type=int, default=4, help='input batch size')
    parser.add_argument('--batch_size', '-b', type=int, default=4, help='input batch size')
    parser.add_argument('--epochs', '-e', type=int, default=10, help='number of epochs')
    parser.add_argument('--lr', '-l', type=float, default=0.001, help='learning rate')
    parser.add_argument('--data_path', '-d', type=str, default='data/.npy', help='path to dataset')
    parser.add_argument('--model_path', '-m', type=str, default='model', help='model path')
    parser.add_argument('--log_path', '-lp', type=str, default='log', help='log path')
    args = parser.parse_args()
    return args


def split_train_and_validate_set(data, ratio=0.8):
    train_images, val_images, train_labels, val_labels = train_test_split(
        data.images, data.labels, shuffle=True, train_size=ratio, stratify=data.labels,
    )
    return train_images, val_images, train_labels, val_labels


def plot_confusion_matrix(writer, cm, class_names, epoch):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(20, 20))
    # color map: https://matplotlib.org/stable/gallery/color/colormap_reference.html
    plt.imshow(cm, interpolation='nearest', cmap="Wistia")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    writer.add_figure('confusion_matrix', figure, epoch)


def train(args):
    print(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = Compose([
        lambda x: np.array(x, dtype=np.float32),
        lambda x: x / 255.0,
        torch.tensor,
    ])
    data = dataset.magic_wand_dataset(root_dir=args.data_path, transform=transform)
    model = magic_wand_model.magic_wand_model(len(data.classes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    last_epoch, best_f1_score = 0, 0
    if os.path.exists(os.path.join(args.model_path, "last.pt")):
        checkpoint = torch.load(os.path.join(args.model_path, "last.pt"))
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        model.load_state_dict(checkpoint["model_state_dict"])
        best_f1_score = checkpoint["best_f1_score"]
        last_epoch = checkpoint["epoch"]

    if not os.path.isdir(args.model_path):
        os.makedirs(args.model_path)

    if not os.path.isdir(args.log_path):
        os.makedirs(args.log_path)
    writer = SummaryWriter(log_dir=args.log_path)

    for epoch in range(last_epoch + 1, args.epochs):
        train_images, val_images, train_labels, val_labels = split_train_and_validate_set(data)
        train_set = dataset.train_set(train_images, train_labels)
        val_set = dataset.val_set(val_images, val_labels)
        # print(train_set.labels.unique(return_counts=True))
        # print(val_set.labels.unique(return_counts=True))
        train_dataloader = DataLoader(dataset=train_set,
                                      num_workers=args.num_workers,
                                      batch_size=args.batch_size,
                                      drop_last=True,
                                      shuffle=True,
                                      )
        val_dataloader = DataLoader(dataset=val_set,
                                    num_workers=args.num_workers,
                                    batch_size=args.batch_size,
                                    drop_last=False,
                                    shuffle=False,
                                    )

        model.train()
        train_loss = []
        progress_bar = tqdm(train_dataloader, colour='green')
        for i, (images, labels) in enumerate(progress_bar):
            labels = labels.long()
            images = images.unsqueeze(1)
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            mean_loss = np.mean(train_loss)
            writer.add_scalar('Train/Loss', mean_loss, epoch * len(train_dataloader) + i)
            progress_bar.set_description(f"Train epoch {epoch} - Loss {mean_loss:.4f}")

        model.eval()
        val_loss = []
        all_labels, all_predictions = [], []
        progress_bar = tqdm(val_dataloader, colour='blue')
        for images, labels in progress_bar:
            with torch.no_grad():
                labels = labels.long()
                images = images.unsqueeze(1)
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)

                loss = criterion(outputs, labels)
                val_loss.append(loss.item())

            all_labels.extend(labels.tolist())
            all_predictions.extend(val_set.one_hot_to_label(outputs).tolist())

        mean_loss = np.mean(val_loss)
        conf_matrix = confusion_matrix(all_labels, all_predictions)
        f1_sc = f1_score(all_labels, all_predictions, average="macro")

        writer.add_scalar("Val/F1", f1_sc, epoch)
        writer.add_scalar("Val/Loss", mean_loss, epoch)
        plot_confusion_matrix(writer, conf_matrix, data.classes, epoch)
        print(f"Epoch {epoch} - Test Loss: {mean_loss:0.4f}")

        checkpoint = {
            "epoch": epoch,
            "best_f1_score": best_f1_score,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        torch.save(checkpoint, os.path.join(args.model_path, "last.pt"))
        if f1_sc > best_f1_score:
            best_f1_score = f1_sc
            torch.save(checkpoint, os.path.join(args.model_path, "best.pt"))


if __name__ == '__main__':
    args = get_args()
    train(args)
