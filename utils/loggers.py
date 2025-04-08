from all_config import train_config
import pandas as pd

class trainLogging:
    def __init__(self, config = train_config()):
        self.config = config
        columns = ['epoch', 'train_loss', 'val_loss']

        self.logs : pd.Dataframe = pd.DataFrame(columns = columns)

    def add_logs(self, epoch, train_logs, val_logs):
        epoch_row = {'epoch': epoch}

        epoch_row.update({f'train_{key}':value for key,value in train_logs.items()})
        epoch_row.update({f'val_{key}':value for key,value in val_logs.items()})

        self.logs.loc[len(self.logs)] =  pd.Series(epoch_row)

    def save_logs(self, filename: str = None):
        self.logs = self.logs.sort_values(by = 'epoch')
        self.logs.to_csv(filename, index = False)