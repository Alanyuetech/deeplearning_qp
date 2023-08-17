import os
import pandas as pd
import shutil
from tabulate import tabulate
from copy_files import copy_files as cf



df = cf(src_dir=os.path.join('..','..', '投资顾问--分析建议'), dst_dir=os.path.join('..', '投资顾问--全部文件')
                , file_formats=['docx'], file_names=['岳靖渊'],exact_match=False)   
 
# print(df.to_string(index=False))
print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))

#增加模块，可以将文件按照文件名的前缀进行分类
# df = cf(src_dir=os.path.join('..', '投资顾问--分析建议'), dst_dir=os.path.join('..', '投资顾问--全部文件')
