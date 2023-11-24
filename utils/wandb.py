import wandb
import numpy as np


class WandbManager:
    def __init__(self, config):
        self.config = config
        self.run = None

    def init(self):
        self.run = wandb.init(project=self.config['project'], 
                              name=self.config['run_name'], 
                              config=self.config)

    def log(self, data):
        self.run.log(data)

    def finish(self):
        self.run.finish()


class WandbTable:
    """
    Example:
        # Create a new WandbTable
        my_table = WandbTable()

        # Add data to the table
        my_table.add_data({"column1": "value1", "column2": "value2", "column3": "value3"})
        my_table.add_data({"column1": "value4", "column2": "value5", "column4": "value6"})

        # Log the table
        run.log({"my-table": my_table.get_table()})
    """
    def __init__(self):
        self.data = []
        self.columns = set()
        self.table = None

    def add_data(self, data_dict):
        self.data.append(data_dict)
        self.columns.update(data_dict.keys())

    def get_table(self):
        self.table = wandb.Table(columns=list(self.columns))
        for data_dict in self.data:
            row = [data_dict.get(col, None) for col in self.columns]
            self.table.add_data(*row)
        return self.table

def wandb_image(img):
    if img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))

    img = wandb.Image(img)
    return img