import pathlib
import os
from matplotlib import ticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_loss(df, best_test_loss):
    min_val_loss_ix = np.argmin(df['val_loss'])

    fig, ax = plt.subplots()
    x_dat = np.arange(len(df))
    ax.plot(x_dat, df['loss'], '#404040', label="train loss", linewidth=0.6)
    ax.plot(x_dat, df['val_loss'], 'g', label="val loss", linewidth=0.6)
    
    ax.plot(min_val_loss_ix, df['loss'].iloc[min_val_loss_ix], '#404040', marker="*")
    ax.plot(min_val_loss_ix, df['val_loss'].iloc[min_val_loss_ix], 'g*')
    ax.plot(min_val_loss_ix, best_test_loss, 'r*', label="test loss")

    ax.plot(x_dat[0:min_val_loss_ix+1], np.repeat(df['val_loss'].iloc[min_val_loss_ix], min_val_loss_ix+1), 'g--', linewidth=0.4)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_xlim(0, len(df))
    ax.set_yscale('log')
    ax.set_ylim(df['loss'].iloc[min_val_loss_ix]-0.05, best_test_loss+1.0)
    ax.get_yaxis().set_major_formatter(ticker.LogFormatter())
    ax.yaxis.set_minor_formatter(ticker.LogFormatter(minor_thresholds=(2, 0.4)))

  
    leg = ax.legend(loc='upper left', fontsize="10")
    for legobj in leg.legendHandles:
      legobj.set_linewidth(1.0)
      legobj.set_marker("*")

    fig.set_size_inches(4.5, 3.5)

    print ("Best model train loss =", df['loss'].iloc[min_val_loss_ix])
    print ("Best model val loss =", df['val_loss'].iloc[min_val_loss_ix])
    print ("Best model test loss =", best_test_loss)


def plot_acc(df, best_test_acc=None):
    min_val_loss_ix = np.argmin(df['val_loss'])

    fig, ax = plt.subplots()
    x_dat = np.arange(len(df))
    ax.plot(x_dat, df['acc'], '#404040', label="train acc", linewidth=0.6)
    ax.plot(x_dat, df['val_acc'], 'g', label="val acc", linewidth=0.6)
    
    ax.plot(min_val_loss_ix, df['acc'].iloc[min_val_loss_ix], '#404040', marker="*")
    ax.plot(min_val_loss_ix, df['val_acc'].iloc[min_val_loss_ix], 'g*')
    # ax.plot(min_val_loss_ix, best_test_acc, 'r*', label="test acc")

    ax.plot(x_dat[0:min_val_loss_ix+1], np.repeat(df['val_acc'].iloc[min_val_loss_ix], min_val_loss_ix+1), 'g--', linewidth=0.4)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(0, len(df))
  
    leg = ax.legend(loc='lower right', fontsize="10")
    for legobj in leg.legendHandles:
      legobj.set_linewidth(1.0)
      legobj.set_marker("*")

    fig.set_size_inches(4.5, 3.5)

    print ("Best model train acc =", df['acc'].iloc[min_val_loss_ix])
    print ("Best model val acc =", df['val_acc'].iloc[min_val_loss_ix])
    # print ("Best model test acc =", best_test_acc)
    

if __name__ == "__main__":
    FILE_PATH = pathlib.Path(__file__).absolute()
    FILE_FOLDER = FILE_PATH.parent
    SHEET_ID = "state_fno1d_lift"
    BEST_TEST_LOSS = 0.08
    
    df = pd.read_excel(os.path.join(FILE_FOLDER,"training_metrics.xlsx"), sheet_name=SHEET_ID)
    plot_loss(df, BEST_TEST_LOSS)
    # plot_acc(df)
    
    plt.tight_layout()
    plt.show()