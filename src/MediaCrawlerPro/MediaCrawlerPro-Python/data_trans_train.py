import pymysql
import os
import re
from config.base_config import KEYWORDS, PLATFORM
import argparse
import asyncio
import sys

# 数据库连接配置
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '11ch06loeilsiE',
    'database': 'media_crawler',
    'charset': 'utf8mb4',
    #'cursorclass': pymysql.cursors.DictCursor,
}
output_dir = './crawled_data'
platform_table_map = {
    'zhihu' : ['zhihu_comment', 'zhihu_content'],
    'xhs' : ['xhs_note', 'xhs_note_comment'],
    'bili' : ['bilibili_video', 'bilibili_video_comment'],
    'wb' : ['weibo_note', 'weibo_note_comment']
}
table_col_map = {
    'zhihu_comment' : 'content',
    'zhihu_content' : 'content_text',
    'xhs_note' : '`desc`',
    'xhs_note_comment' : 'content',
    'bilibili_video': '`desc`',
    'bilibili_video_comment': 'content',
    'weibo_note': 'content',
    'weibo_note_comment': 'content'
}

def parse_cmd():
    parser = argparse.ArgumentParser(description="Arguments of train data from sql to txt.")
    #parser.add_argument('--platform', type=str, help="Enter platform names(separated by , )",
    #                    choices=["xhs", "zhihu", "bili", "wb"], default=PLATFORM)
    parser.add_argument('--platform', type=str, help="Enter platform names(separated by , )",
                       default=PLATFORM)
    parser.add_argument('--separator', type=str, help="Enter separator",
                       default='//////')
    args = parser.parse_args()
    return args

def reg_filter(text):
    #pattern = re.compile(r'.*医院.*|.*医生.*')
    #pattern = re.compile(r'.*')
    #pattern = re.compile(r'.*医.*')
    pattern = re.compile(r'.*医院.*')
    return pattern.search(text)

def transform(platforms, sep):
    conn = pymysql.connect(**db_config)
    try:
        with conn.cursor() as cursor:
            for platform in platforms:
                table_names = platform_table_map[platform]
                for table_name in table_names:
                    num = 0
                # 构造 SQL 查询语句
                    col_name = table_col_map[table_name]
                    sql = f"SELECT {col_name} FROM {table_name};"
                    print(sql)
                    cursor.execute(sql)

                    # 获取查询结果
                    results = cursor.fetchall()
                    file_name = f"{table_name}.txt"
                    file_path = os.path.join(output_dir, platform, file_name)
                    # 打开一个本地文本文件以写入数据
                    with open(file_path, 'w', encoding='utf-8') as file:
                        #file.write(sep.join(map(str, results)))
                        output = ''
                        for row in results:
                            text = row[0]
                            if reg_filter(text):
                                output += text + sep
                                num += 1
                            # 写入文件，并添加换行符
                        output.rstrip(sep)
                        #print(output)
                        file.write(output)
                    '''for row in results:
                        # 将每行数据用指定的分隔符连接
                        # output += row[0] + sep
                        # num += 1
                        text = row[0]
                        if reg_filter(text):
                            #print(text)
                            num += 1'''
                    print(f"{num}条数据已成功写入{file_path}文件")
    finally:
        conn.close()

if __name__ == '__main__':
    #set_pattern(KEYWORDS)
    args = parse_cmd()
    #transform(['zhihu'], '//////')
    platforms = args.platform.split(',')
    sep = args.separator
    transform(platforms, sep)



