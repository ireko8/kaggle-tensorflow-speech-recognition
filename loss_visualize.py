from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == "__main__":

    path = Path("cv/STFTCNN/2017_12_12_01_45_32")
    history = pd.read_csv(path/"fold_0_log.csv")
    plt.plot(history.epoch, history.acc, label="train_acc")
    plt.plot(history.epoch, history.val_acc, label="val_acc")
    plt.legend()
    plt.savefig(path/"visualize_loss.pdf")
