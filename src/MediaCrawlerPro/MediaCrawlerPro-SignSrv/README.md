## 免责声明

本仓库的所有内容仅供学习使用，禁止用于商业用途。任何人或组织不得将本仓库的内容用于非法用途或侵犯他人合法权益。

我们提供的爬虫仅能获取抖音、快手、哔哩哔哩、小红书、百度贴吧、微博平台上**公开的信息**，

我们强烈反对任何形式的隐私侵犯行为。如果你使用本项目进行了侵犯他人隐私的行为，我们将与你保持距离，并支持受害者通过法律手段维护自己的权益。<br>

对于因使用本仓库内容而引起的任何法律责任，本仓库不承担任何责任。使用本仓库的内容即表示您同意本免责声明的所有条款和条件<br>

## MediaCrawlerSignSrv 平台请求签名服务
将请求签名的功能从MediaCrawler中独立出来，作为一个独立的服务，方便调用。
另外，这个服务也可以作为一个独立的服务，供其他项目调用。

## 项目部署安装

### 本地安装
> python推荐版本：3.9.6， requirements.txt中的依赖包是基于这个版本的，其他版本可能会有依赖装不上问题。
> 
> 本地安装签名服务时，需要nodejs环境，版本大于等于16以上

#### 1、新建Pro版本目录
```shell
# 新建目录MediaCrawlerPro并进入
mkdir MediaCrawlerPro
cd MediaCrawlerPro
```

##### 2、克隆签名服务仓库并安装依赖
```shell
# 先克隆签名服务仓库并安装依赖
git clone https://github.com/MediaCrawlerPro/MediaCrawlerPro-SignSrv
cd MediaCrawlerPro-SignSrv

# 创建虚拟环境并安装签名服务的依赖，
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

##### 3、启动签名服务
> 本地安装签名服务时，需要nodejs环境，版本大于等于16以上
```shell
python app.py 
```

### Docker安装
```shell
docker build -t mediacrawler_signsrv .
docker run -p 8989:8989 -e LOGGER_LEVEL=ERROR mediacrawler_signsrv
```

## 项目目录结构说明
```
MediaCrawlerPro-SignSrv 目录结构说明
├── apis                          # API接口目录  
│   ├── base_handler.py           # Tornado Handler的基础类  
│   ├── douyin.py                 # 抖音签名API接口  
│   └── xiaohongshu.py            # 小红书签名API接口  
├── constant                      # 常量定义目录  
│   ├── base_constant.py          # 基础常量  
│   └── error_code.py             # 错误代码定义  
├── logic                         # 业务签名逻辑目录  
│   ├── douyin                    # 抖音签名业务逻辑  
│   │   └── douyin_logic.py       # 抖音签名业务逻辑实现  
│   └── xhs                       # 小红书签名业务逻辑  
│       ├── help.py               # 小红书辅助函数  
│       └── xhs_logic.py          # 小红书签名业务逻辑实现  
├── params                        # 参数定义目录  
│   ├── base_model.py             # 基础参数模型  
│   ├── douyin_sign.py            # 抖音签名相关参数  
│   └── xiaohongshu_sign.py       # 小红书签名相关参数  
├── pkg                           # 项目包目录  
│   ├── custom_exceptions         # 自定义异常目录  
│   │   └── base_exceptions.py    # 基础异常类  
│   ├── js                        # JavaScript目录  
│   │   ├── douyin.js             # 抖音相关JS文件  
│   │   ├── stealth.min.js        # 去除浏览器自动化特征的JS  
│   │   └── xhs.js                # 小红书相关JS文件  
│   ├── playwright                # Playwright目录  
│   │   ├── douyin_manager.py     # 抖音Playwright管理器  
│   │   ├── manager.py            # 通用Playwright管理器  
│   │   └── xhs_manager.py        # 小红书Playwright管理器  
│   └── utils                     # 工具函数目录  
│       ├── base_utils.py         # 基础工具函数  
│       └── crawler_util.py       # 爬虫工具函数  
├── Dockerfile                    # Docker配置文件  
├── LICENSE                       # 开源协议  
├── README.md                     # 项目说明文档  
├── app.py                        # 应用程序入口  
├── config.py                     # 配置文件  
├── context_vars.py               # 上下文变量定义    
├── requirements.txt              # 依赖项说明文件  
└── urls.py                       # URL路由配置
```