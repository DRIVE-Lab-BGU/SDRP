import os
import csv

class Log(object):
    def __init__(self, csv_file_name, mode='a'):
        # base_path = os.path.dirname(os.path.abspath(__file__))
        # csv_file_name = os.path.join(base_path, 'logs', csv_file_name)
        self.csv_file_name = csv_file_name
        self.mode = mode
        pass

    def logData(self, data: dict, stds: dict):
        # Ensure both dicts have the same keys
        if data.keys() != stds.keys():
            raise ValueError("Keys in data and stds dictionaries do not match!")

        # Check if we need to write the header
        write_header = not os.path.exists(self.csv_file_name)

        # Open the file
        csv_file = open(self.csv_file_name, mode=self.mode, newline='')
        writer = csv.DictWriter(csv_file, fieldnames=["epoch", "eval_reward", "eval_std"])

        # Only write header if file is new and mode is append
        if write_header and self.mode == 'a':
            writer.writeheader()

        for key in data:
            row = {
                "epoch": key,
                "eval_reward": data[key],
                "eval_std": stds[key]
            }
            writer.writerow(row)

        csv_file.close()


        # for _, (key, value) in enumerate(data.items(), start=0):
        #     row = {
        #         'epoch': key,
        #         'eval_reward': value
        #     }
        #     writer.writerow(row)


