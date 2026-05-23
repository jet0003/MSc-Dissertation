import matplotlib.pyplot as plt
import seaborn as sns


def plot_metric_history(history, title="Training History"):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.legend()
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()


def plot_bar(values, labels, title):
    plt.figure(figsize=(8, 5))
    sns.barplot(x=labels, y=values)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
