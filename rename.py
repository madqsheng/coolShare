import pandas as pd
import os
import subprocess


# 指定 Excel 文件路径
excel_file_path = r'D:\WeChat\WeChat Files\WeChat Files\wxid_xuo041oyae8e22\FileStorage\File\2023-12\微信主体命名.xlsx'
#截图文件目录
file_dir = r'D:\卢艳\父亲节微信主体截图'

# 从文件名中提取数字部分
def extract_number_from_filename(filename):
    numbers = filename.split('.')[0]
    return int(numbers)

# 定义自定义排序函数，使用提取的数字作为排序关键字
def custom_sort(filename): 
    return extract_number_from_filename(filename)

#获取所有文件名
filenames = os.listdir(file_dir)

#过滤掉已经改名的
filter_file_list=[]
for filename in filenames:
    base_name, extension = os.path.splitext(filename)
    if base_name.isdigit():
        # 将符合条件的文件名添加到列表
        filter_file_list.append(filename)

# 使用自定义排序函数对文件名列表进行排序
sorted_filter_filenames = sorted(filter_file_list, key=custom_sort)
sorted_filter_file_list=[filename for filename in sorted_filter_filenames]



print(len(sorted_filter_file_list))

# 指定要提取的列名
column_name_to_extract = '公司名称'  # 替换为你要提取的列的实际名称
# 读取 Excel 文件
df = pd.read_excel(excel_file_path)
# 提取指定列的所有内容
column_data = df[column_name_to_extract]

column_data=list(column_data)
# print(column_data)

begin=8

#打印提取的列数据
for i,new_name in enumerate(column_data[begin*50:(begin+1)*50],start=begin*50):
    new_name.replace('\n', '')
    old_name = sorted_filter_file_list[i-50*begin]
    new_name= new_name+'.jpg'

    old_name_path = os.path.join(file_dir,old_name)
    new_name_path = os.path.join(file_dir,new_name)
    print(old_name,new_name)

    # 执行文件重命名
    try:
        os.rename(old_name_path, new_name_path)
        # print(f"文件重命名成功：'{old_name}' 已更名为 '{new_name}'")
    except FileNotFoundError:
        print(f"文件 '{old_name_path}' 不存在")
    except FileExistsError:
        print(f"文件 '{new_name_path}' 已经存在")
    except Exception as e:
        print(f"发生错误：{e}")

import tesserocr
tesserocr.image_to_text