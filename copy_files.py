import os
import pandas as pd
import shutil

def copy_files(src_dir, dst_dir, file_formats=[], file_names=[], exact_match=False):
    data = {'filename': [], 'original_path': [], 'file_size(KB)': []}
    is_empty_format = len(file_formats) == 0
    is_empty_name = len(file_names) == 0
    found_file = False

    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    for dirpath, dirnames, filenames in os.walk(src_dir):
        for filename in filenames:
            file_format = filename.split('.')[-1]
            file_name = filename.split('.')[0]

            # Check whether file name matches one of the file_names
            name_match = False
            if not is_empty_name:
                for name in file_names:
                    if (exact_match and name == file_name) or (not exact_match and name in file_name):
                        name_match = True
                        break

            if (is_empty_format or file_format in file_formats) and (is_empty_name or name_match):
                found_file = True
                src_file_path = os.path.join(dirpath, filename)
                dst_file_path = os.path.join(dst_dir, filename)
                shutil.copy2(src_file_path, dst_file_path)

                file_size = os.path.getsize(src_file_path) / 1024  # size in KB
                data['filename'].append(filename)
                data['original_path'].append(dirpath)
                data['file_size(KB)'].append(file_size)

    if not found_file:
        if is_empty_format and is_empty_name:
            print("无符合条件的文件")
        elif is_empty_format:
            print("无以下名称的文件：" + ', '.join(file_names))
        elif is_empty_name:
            print("无以下格式的文件：" + ', '.join(file_formats))

    return pd.DataFrame(data)


