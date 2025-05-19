# 声明：本代码仅供学习和研究目的使用。使用者应遵守以下原则：  
# 1. 不得用于任何商业用途。  
# 2. 使用时应遵守目标平台的使用条款和robots.txt规则。  
# 3. 不得进行大规模爬取或对平台造成运营干扰。  
# 4. 应合理控制请求频率，避免给目标平台带来不必要的负担。   
# 5. 不得用于任何非法或不当的用途。
#   
# 详细许可条款请参阅项目根目录下的LICENSE文件。  
# 使用本代码即表示您同意遵守上述原则和LICENSE中的所有条款。  


import argparse

import config
import constant
from pkg.tools import utils


async def parse_cmd():
    # 读取command arg
    parser = argparse.ArgumentParser(description='Media crawler program.')
    parser.add_argument('--task_id', type=int, default=config.TASK_ID)
    parser.add_argument('--username', type=str, default=config.USERNAME)
    parser.add_argument('--platform', type=str, help='Media platform select (xhs | dy | ks | bili | wb | tieba | zhihu)',
                        choices=[
                            constant.XHS_PLATFORM_NAME,
                            constant.DOUYIN_PLATFORM_NAME,
                            constant.KUAISHOU_PLATFORM_NAME,
                            constant.WEIBO_PLATFORM_NAME,
                            constant.BILIBILI_PLATFORM_NAME,
                            constant.TIEBA_PLATFORM_NAME,
                            constant.ZHIHU_PLATFORM_NAME
                        ], default=config.PLATFORM)
    parser.add_argument('--type', type=str, help='crawler type (search | detail | creator)',
                        choices=["search", "detail", "creator"], default=config.CRAWLER_TYPE)
    #parser.add_argument('--task_id', type=str, default='0')
    #parser.add_argument('--start_time', type=str, default=config.START_TIME)
    #parser.add_argument('--end_time', type=str, default=config.END_TIME)
    parser.add_argument('--max_note', type=int, default=config.CRAWLER_MAX_NOTES_COUNT)
    parser.add_argument('--max_comment', type=int, default=config.PER_NOTE_MAX_COMMENTS_COUNT)
    parser.add_argument('--if_level', type=int, default=0)
    parser.add_argument('--keywords', type=str,
                        help='please input keywords', default=config.KEYWORDS)
    #parser.add_argument('--all_names', type=str,
    #                    help='please input keywords', default=config.KEYWORDS)
    parser.add_argument('--start', type=int,
                        help='number of start page', default=config.START_PAGE)
    parser.add_argument('--save_data_option', type=str,
                        help='where to save the data (csv or db or json)', choices=['csv', 'db', 'json'],
                        default=config.SAVE_DATA_OPTION)

    args = parser.parse_args()

    utils.logger.info(f"!!!传入MediaCrawlerPro的参数为：")
    utils.logger.info(f"args.task_id:{args.task_id}{ type(args.task_id)}\n args.max_note:{args.max_note}{ type(args.max_note)} \n args.max_comment:{args.max_comment}\n args.if_level:{args.if_level}")
    # override config
    config.TASK_ID = args.task_id
    config.USERNAME = args.username
    config.PLATFORM = args.platform
    config.CRAWLER_TYPE = args.type
    config.START_PAGE = args.start
    config.KEYWORDS = args.keywords
    config.CRAWLER_MAX_NOTES_COUNT = args.max_note
    config.PER_NOTE_MAX_COMMENTS_COUNT = args.max_comment
    config.ENABLE_GET_SUB_COMMENTS = True if args.if_level == 1 else False
    return args
