#定义一个类，用于获取数据库或者API接口的数据。设置内存阈值，使获取到的数据最大限度利用内存阈值。如果超过阈值，先通过数据内存和内存之阈值的比值确定保留多少数据条数。如果通过比值计算出来的数据条数还是超过阈值，再通过其他方法削减数据。
import pandas as pd
import numpy as np

class DataFetcher:
    def __init__(self, data, batch_memory_limit_mb=500):
        self.data = data
        self.batch_memory_limit_mb = batch_memory_limit_mb
        self.prev_batch_rows = None

    def get_batches(self):
        total_rows = len(self.data)
        total_memory_mb = self.data.memory_usage().sum() / 1024 ** 2

        print(f"Initial total memory usage: {total_memory_mb} MB")

        # Set initial number of rows in a batch
        batch_rows = total_rows // 2 if self.prev_batch_rows is None else self.prev_batch_rows

        while len(self.data) > 0:
            batch_data = self.data.iloc[:batch_rows]
            batch_memory_mb = batch_data.memory_usage().sum() / 1024 ** 2

            if batch_memory_mb <= self.batch_memory_limit_mb:
                remaining_data = self.data.iloc[batch_rows:]
                if len(remaining_data) <= batch_rows:
                    increased_batch_rows = int(batch_rows * self.batch_memory_limit_mb / batch_memory_mb)
                    increased_batch_data = self.data.iloc[:increased_batch_rows]
                    increased_batch_memory_mb = increased_batch_data.memory_usage().sum() / 1024 ** 2

                    if increased_batch_memory_mb <= self.batch_memory_limit_mb:
                        batch_data, batch_rows, batch_memory_mb = increased_batch_data, increased_batch_rows, increased_batch_memory_mb

                self.prev_batch_rows = batch_rows
                print(f"Returning data batch with memory usage: {batch_memory_mb} MB")
                yield batch_data
                self.data = self.data.iloc[batch_rows:]
            else:
                batch_rows //= 2  # decrease the batch size


# 使用方式：df为10**5行，10**5列,全为1的数据
n_col_row = 2*(10**4)
df = pd.DataFrame(np.ones((n_col_row, n_col_row)))

data_loader = DataFetcher(df)


import time
start_time = time.time()
for batch in data_loader.get_batches():
    # Do something with the batch
    pass
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")
