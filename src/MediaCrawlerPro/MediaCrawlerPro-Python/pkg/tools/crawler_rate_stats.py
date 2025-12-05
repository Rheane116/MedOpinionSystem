# 声明：本代码仅供学习和研究目的使用。使用者应遵守以下原则：  
# 1. 不得用于任何商业用途。  
# 2. 使用时应遵守目标平台的使用条款和robots.txt规则。  
# 3. 不得进行大规模爬取或对平台造成运营干扰。  
# 4. 应合理控制请求频率，避免给目标平台带来不必要的负担。   
# 5. 不得用于任何非法或不当的用途。
#   
# 详细许可条款请参阅项目根目录下的LICENSE文件。  
# 使用本代码即表示您同意遵守上述原则和LICENSE中的所有条款。  


import time
from typing import Dict, Optional
from pkg.tools import utils


class CrawlerRateStats:
    """爬虫速率统计类"""
    
    def __init__(self):
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.video_count: int = 0
        self.comment_count: int = 0
        self.note_count: int = 0
        self.creator_count: int = 0
        self.total_count: int = 0
        self.platform: str = ""
        self.keyword: str = ""

        self.max_note:int = 0
        self.max_comment:int = 0
        self.if_level:bool = False
        
    def start_crawling(self, platform: str, keyword: str = "", max_note:int=0, max_comment:int=0, if_level:bool=False):
        """开始爬虫统计"""
        self.start_time = time.time()
        self.platform = platform
        self.keyword = keyword
        self.video_count = 0
        self.comment_count = 0
        self.note_count = 0
        self.creator_count = 0
        self.total_count = 0
        self.max_note = max_note
        self.max_comment = max_comment
        self.if_level = if_level
        utils.logger.info(f"[CrawlerRateStats] 开始爬虫统计 - 平台: {platform}, 关键词: {keyword}")
        
    def end_crawling(self):
        """结束爬虫统计"""
        self.end_time = time.time()
        utils.logger.info(f"[CrawlerRateStats] 结束爬虫统计")
        
    def increment_video_count(self, count: int = 1):
        """增加视频数量"""
        self.video_count += count
        self.total_count += count
        utils.logger.info(f"[CrawlerRateStats] 视频数量: {self.video_count}")
        
    def increment_note_count(self, count: int = 1):
        """增加帖子数量"""
        self.note_count += count
        self.total_count += count
        utils.logger.info(f"[CrawlerRateStats] 帖子数量: {self.note_count}")

    def increment_comment_count(self, count: int = 1):
        """增加评论数量"""
        self.comment_count += count
        self.total_count += count
        utils.logger.info(f"[CrawlerRateStats] 评论数量: {self.comment_count}")
        
    def increment_creator_count(self, count: int = 1):
        """增加创作者数量"""
        self.creator_count += count
        self.total_count += count
        utils.logger.info(f"[CrawlerRateStats] 创作者数量: {self.creator_count}")
        
    def get_elapsed_time(self) -> float:
        """获取已用时间（秒）"""
        if self.start_time is None:
            return 0.0
        end_time = self.end_time if self.end_time else time.time()
        return end_time - self.start_time
        
    def get_crawling_rate(self) -> float:
        """获取爬虫速率（条/秒）"""
        elapsed_time = self.get_elapsed_time()
        if elapsed_time <= 0:
            return 0.0
        return self.total_count / elapsed_time
        
    def get_stats_summary(self) -> Dict:
        """获取统计摘要"""
        elapsed_time = self.get_elapsed_time()
        rate = self.get_crawling_rate()
        
        return {
            "platform": self.platform,
            "keyword": self.keyword,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "elapsed_time": elapsed_time,
            "video_count": self.video_count,
            "note_count": self.note_count,
            "comment_count": self.comment_count,
            "creator_count": self.creator_count,
            "total_count": self.total_count,
            "crawling_rate": rate,
        }
        
    def print_final_stats(self):
        """打印最终统计结果"""
        stats = self.get_stats_summary()
        
        utils.logger.info("\n" + "="*60)
        utils.logger.info("爬虫速率统计报告")
        utils.logger.info("="*60)
        utils.logger.info(f"平台: {stats['platform']}")
        utils.logger.info(f"关键词: {stats['keyword']}")
        utils.logger.info(f"最大爬取帖子/视频数量：{self.max_note}")
        utils.logger.info(f"最大爬取评论数量：{self.max_comment}")
        utils.logger.info(f"是否爬取二级评论：{self.if_level}")
        utils.logger.info(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stats['start_time'])) if stats['start_time'] else 'N/A'}")
        utils.logger.info(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stats['end_time'])) if stats['end_time'] else 'N/A'}")
        utils.logger.info(f"总耗时: {stats['elapsed_time']:.2f} 秒")
        utils.logger.info(f"视频数量: {stats['video_count']}")
        utils.logger.info(f"帖子数量: {stats['note_count']}")
        utils.logger.info(f"评论数量: {stats['comment_count']}")
        utils.logger.info(f"创作者数量: {stats['creator_count']}")
        utils.logger.info(f"总数据量: {stats['total_count']}")
        utils.logger.info(f"爬虫速率: {stats['crawling_rate']:.2f} 条/秒")
        utils.logger.info("="*60)
        

# 全局爬虫统计实例
#crawler_stats = CrawlerRateStats()
