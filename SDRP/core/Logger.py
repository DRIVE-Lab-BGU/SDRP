import os
import csv

class Log(object):
    def __init__(self, csv_file_name, mode='w'):
        # base_path = os.path.dirname(os.path.abspath(__file__))
        # csv_file_name = os.path.join(base_path, 'logs', csv_file_name)
        self.csv_file_name = csv_file_name
        self.mode = mode
        pass

    def logData(self, data):

        write_header = not os.path.exists(self.csv_file_name)
        csv_file = open(self.csv_file_name, mode=self.mode, newline='')
        writer = csv.DictWriter(csv_file, fieldnames=["epoch", "eval_reward"])
        if write_header and self.mode == 'a':
            writer.writeheader()

        for _, (key, value) in enumerate(data.items(), start=0):
            row = {
                'epoch': key,
                'eval_reward': value
            }
            writer.writerow(row)
        csv_file.close()

