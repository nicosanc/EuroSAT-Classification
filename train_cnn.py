from model.ConvNet import ConvNet
from torch.nn.functional import softmax
import torch
import pathlib
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn import metrics
from matplotlib import gridspec
import matplotlib.pyplot as plt
from prepare_dataset import extract_data, prepare_data, EuroSATDataset, load_dataset, load_dataloaders
from visualize_data import visualize_data, denormalize_img, class_distribution

def save_checkpoint(model, epoch, stats, file):
    state = {
      'epoch': epoch,
      'state_dict': model.state_dict(),
      'stats': stats
    }

    filename = pathlib.Path(file)
    torch.save(state, filename)

def early_stopping(stats, curr_patience, prev_valid_loss):
    if stats[-1][1] > prev_valid_loss:
        curr_patience += 1
    else:
        curr_patience = 0
        prev_valid_loss = stats[-1][1]
    return curr_patience, prev_valid_loss

def train_epoch(data_loader, model, criterion, optimizer):
    model.cuda()
    model.train()
    for i, (X, y) in enumerate(data_loader):
        X, y = X.cuda(), y.cuda()
        # Clear the gradients
        optimizer.zero_grad()
        # Load up the model with the training data loader
        output = model(X)
        # Calculate the loss using cross entropy loss func
        loss = criterion(output, y)
        # Backpropagate
        loss.backward()
        optimizer.step()
        if i % 1000 == 0:
            print(f"Training batch loss {loss}")

def predictions(logits):
    pred = torch.argmax(logits, dim = 1)
    return pred

def eval_epoch(
    train_load,
    valid_load,
    model,
    criterion,
    epoch,
    stats,
    test_load = None,
    update_plot=True):

    def get_metrics(load):
        model.eval()
        y_true, y_pred, y_score = [], [], []
        correct, total = 0, 0
        running_loss = []
        for i, (X, y) in enumerate(load):
            with torch.no_grad():
          # our project leverages the GPU offered by colab so we're going to
          # set the data to work with cuda
                X, y = X.cuda(), y.cuda()
                output = model(X)
                predicted = predictions(output.data)
                y_true.append(y)
                y_pred.append(predicted)
                y_score.append(softmax(output.data, dim = 1))
                total += len(y)
                correct += (predicted == y).sum().item()
                running_loss.append(criterion(output, y).item())
        y_true = torch.cat(y_true)
        y_pred = torch.cat(y_pred)
        y_score = torch.cat(y_score)
        loss = np.mean(running_loss)
        accuracy = correct / total
        return accuracy, loss, y_true, y_score

    train_accuracy, train_loss, _, _ = get_metrics(train_load)
    print(f"epoch {epoch}, {train_accuracy}, {train_loss}")
    valid_accuracy, valid_loss,_, _ = get_metrics(valid_load)
    print(f"epoch {epoch}, {valid_accuracy}, {valid_loss}")
    epoch_stats = [
        valid_accuracy,
        valid_loss,
        train_accuracy,
        train_loss,
    ]

    if test_load:
        epoch_stats += get_metrics(test_load)
        y_true, y_score = epoch_stats[-2], epoch_stats[-1]
        y_pred = torch.argmax(y_score, dim = 1)

        conf_mat = metrics.confusion_matrix(y_true.cpu().numpy(), y_pred.cpu().numpy())
        plt.figure(figsize=(10,8))
        sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues")
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.show()

        class_report = metrics.classification_report(y_true.cpu().numpy(), y_pred.cpu())
        print(class_report)

    stats.append(epoch_stats)


if __name__ == "__main__":

    model = ConvNet()
    model.cuda()
    #model = model.cuda()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = .001, weight_decay = .01)

    # initialize some useful variables for evaluating our model
    stats = []
    start_epoch = 0
    patience = 5
    current_patience = 0

    image_paths, labels, train_dataset, valid_dataset, test_dataset = load_dataset(root_dir=os.get_cwd())

    # create a transformations object for our dataset to inherit
    class_count = Counter(labels)
    class_names = list(class_count.keys())
    counts = list(class_count.values())
    plt.pie(counts, labels=class_names, colors=sns.color_palette('pastel'),
            autopct='%.0f%%')
    plt.show()

    visualize_data(train_dataset)
    test_load, train_load, valid_load = load_dataloaders(test_dataset, train_dataset, valid_dataset)

    # To have a baseline validation loss to compare, we evaluate an epoch
    # of the model with random initialization

    eval_epoch(train_load, valid_load, model, criterion, start_epoch, stats
    )

    prev_valid_loss = stats[-1][1]
    print(stats[-1])

    while current_patience < patience:
    # train model for an epoch
        train_epoch(train_load, model, criterion, optimizer)
        eval_epoch(train_load, valid_load, model, criterion, start_epoch, stats)
        if prev_valid_loss > stats[-1][1]:
            save_checkpoint(model, start_epoch, stats, 'best_model.pt')
        current_patience, prev_valid_loss = early_stopping(stats, current_patience, prev_valid_loss)
        start_epoch +=1

    # test the model against the test set
    model.load_state_dict(torch.load('best_model.pt')["state_dict"])
    eval_epoch(train_load, valid_load, model, criterion, start_epoch, stats, test_load)

    test_accuracy, test_loss = stats[-1][-4:-2]
    print(f"Test accuracy: {test_accuracy}")
    print(f"Test loss: {test_loss}")
