# 1. 爬虫框架

- 动机

  - 问题1：为什么会出现爬虫框架？

    **一个爬虫的程序并不单单是向服务器发送请求，解析，存储。**还有异常处理，日志记录，配置和参数，考虑性能和效率，还有并发爬虫，更重要的还有**用户代理，反反爬虫，任务调度**等。框架的基本特征就是**不必从头开始，灵活扩展**。scrapy是爬虫框架里的佼佼者。

  - 问题2：一个完整的爬虫程序通常包括哪些部分：

    1. **起始URL和链接管理：**
       - 确定要抓取的起始URL或URL列表。
       - 管理链接，跟踪要访问的页面，确保不重复抓取相同的页面。
    2. **HTTP请求发送：**
       - 使用HTTP库发送请求到目标网站的URL，获取网页的HTML内容。
       - 设置请求头部，处理Cookie和Session等身份验证。
    3. **网页解析和数据提取：**
       - 使用HTML解析器（如Beautiful Soup或lxml）解析HTML页面，或使用正则表达式等方法提取所需的数据。
       - 根据网页的结构和内容，编写解析规则和选择器（XPath、CSS选择器等）来定位和提取数据。
    4. **数据处理和存储：**
       - 对从网页中提取的数据进行清洗、转换和处理，以确保数据的质量。
       - 将数据存储到适当的位置，如数据库、CSV文件、JSON文件等。
    5. **错误处理和异常处理：**
       - 处理请求过程中可能出现的异常，如连接超时、HTTP错误、解析错误等。
       - 设置重试机制，以应对临时性错误。
    6. **并发和性能优化：**
       - 使用异步请求或多线程/多进程来提高爬取的效率。
       - 控制请求速率，避免对目标网站造成不必要的负载。
    7. **用户代理和反反爬虫：**
       - 设置合适的用户代理（User-Agent）以模拟不同的浏览器行为。
       - 处理反爬虫机制，如限制请求频率、验证码识别等。
    8. **日志记录和监控：**
       - 记录爬虫的运行日志，以便排查问题和监控爬取进度。
       - 可以使用工具或服务来监控爬虫的健康状态。
    9. **定时任务和调度：**
       - 设置爬虫的运行计划，以定期执行抓取任务。
       - 使用调度器来管理爬虫的运行，确保按计划执行。
    10. **配置和参数：**
        - 提供配置文件或参数，以允许用户自定义爬虫的行为，如请求延迟、爬取深度等。
    11. **合规性和伦理：**
        - 遵守网站的Robots协议，尊重网站的隐私和使用政策。
        - 确保爬取任务的合法性和合规性，避免侵犯法律和伦理规定。
    12. **交互界面（可选）：**
        - 可能需要一个交互式界面或Web界面，以便用户控制爬虫的行为和监控任务状态。

# 2. Scrapy简介

## 2.1 安装与运行

- 安装

  1. pip安装：`pip install scrapy`

  2. conda安装：`conda install scrapy`

  3. 源码安装：

     ```bash
     git clone https://github.com/scrapy/scrapy.git
     cd scrapy
     pip install -r requirements.txt
     python setup.py install
     ```

- **命令行运行**：不是常见的`python xx.py `的模式，**scrapy预设了命令**

  1. **创建新的Scrapy项目：**

     `scrapy startproject project_name`

     得到的项目结构：

     ```cmd
     scrapy.cfg         	  # Scrapy 部署时的配置文件
     project_name          # 项目的模块，引入的时候需要从这里引入
         __init__.py    
         items.py     	  # Items 的定义，定义爬取的数据结构
         middlewares.py    # Middlewares 的定义，定义爬取时的中间件
         pipelines.py      # Pipelines 的定义，定义数据管道
         settings.py       # 配置文件
         __init__.py
         spiders           # Spiders 的文件夹,里面会放置核心的spider对象
         	__init__.py
     ```

  2. **创建新的Spider：**

     ```bash
     cd project_name
     scrapy genspider spider_name domain.com
     #在spiders文件夹下生成spider_name.py
     #两个参数会预设spider_name.py里的内容
     ```

     **spider_name.py** 

     ```python
     import scrapy
     
     
     class SpiderNameSpider(scrapy.Spider):
         name = 'spider_name'
         allowed_domains = ['domain.com']
         start_urls = ['http://domain.com/']
     
         def parse(self, response):
             pass
     '''
     预设关系
     参数1.spider_name:
         文件名spider_name.py
         类名：SpiderName
         类属性：name = 'spider_name'
     参数2：domain.com
     	类属性：allowed_domains = ['domain.com']
     	类属性：start_urls = ['http://domain.com/']
     '''
     ```

  3. **运行Spider：**

     `scrapy crawl spider_name`

     - 问题：spider_name是什么？

       A：是**spiders文件夹**下**spider_name.py**里**SpiderNameSpider类**里的**name属性**，其实跟spider_name.py文件名，SpiderNameSpider类名没关系

  4. **列出可用的Spider：**

     `scrapy list`

  5. **检查Spider：**检查Spider的语法错误和其他问题，以确保它能正常运行。

     `scrapy check spider_name`

  6. **Shell交互环境：**用于测试XPath、CSS选择器和调试Spider。

     `scrapy shell`

  7. **导出数据到文件：**支持**JSON、CSV、XML**等格式，**注意数据是通过item.py定义**

     ```bash
     #json
     scrapy crawl spider_name -o output.json
     
     #还可以每一个 Item 输出一行 JSON，输出后缀为 jl，为 jsonline 的缩写
     scrapy crawl spider_name -o quotes.jl
     #或者
     scrapy crawl spider_name -o quotes.jsonlines
     ```

  8. **查看Scrapy版本：**

     `scrapy version`

## 2.2 架构

![scrapy框架](..\示例图片\scrapy框架.jpg)

- Scrapy架构
  - **Engine**：引擎，用来处理整个系统的数据流处理，触发事务，**是整个框架的核心**。
  - **Item**：项目，它定义了爬取结果的数据结构，爬取的数据会被赋值成该对象。
  - **Scheduler**： 调度器，用来接受引擎发过来的请求并加入队列中，并在引擎再次请求的时候提供给引擎。
  - **Downloader**：下载器，用于下载网页内容，并将网页内容返回给蜘蛛。
  - **Spiders**：蜘蛛，其内定义了爬取的逻辑和网页的解析规则，它主要负责解析响应并生成提取结果和新的请求。
  - **Item Pipeline**：项目管道，负责处理由蜘蛛从网页中抽取的项目，它的主要任务是清洗、验证和存储数据。
  - **Downloader Middlewares**：下载器中间件，位于引擎和下载器之间的钩子框架，主要是处理引擎与下载器之间的请求及响应。
  - **Spider Middlewares**：蜘蛛中间件，位于引擎和蜘蛛之间的钩子框架，主要工作是处理蜘蛛输入的响应和输出的结果及新的请求。
- 架构对应项目结构
  - 架构里的8个部分并不会完全对应项目里的结构
    1. **Item对应item.py**
    2. **Spiders对应spiders文件夹下文件里自定义的spider类**
    3. **Item Pipeline对应pipelines.py**
    4. **Downloader Middlewares和Spider Middlewares对应middlewares.py**
    5. **Engine和Scheduler是框架的核心，以及Downloader都没有对应到项目文件，框架源码进行了预设**

## 2.3 运行过程

![scrapy运行](..\示例图片\scrapy运行.png)

1. **Scrapy运行某个Spider（`scrapy crawl spider_name`）：** 引擎（Engine）根据指定的Spider名称启动爬虫。

2. **Engine从Spider中找到要爬取的第一个URL：** 引擎从Spider中获取**初始的URL**，这通常是通过Spider类的`start_urls`属性定义的，或者`start_requests`方法。

3. **Engine通过Scheduler调度URL：** 引擎将初始URL通过Scheduler进行调度，将其包装为`scrapy.Request`的形式并加入请求队列。

   **注意**：*如果**初始的URL**通过Spider类的`start_urls`属性定义，这些URL会自动加入到`scrapy.Request`队列中，成为初始请求。这意味着Spider会自动为每个`start_urls`中的URL创建一个初始的`scrapy.Request`对象并将其加入队列。*

   然而，如果你在Spider中定义了`start_requests`方法，并在该方法中自定义`scrapy.Request`对象并加入队列，那么`start_urls`中定义的URL不会自动加入队列。在这种情况下，只有`start_requests`方法中自定义的`scrapy.Request`对象会成为初始请求队列中的请求。

   换句话说，`start_requests`方法中的自定义请求会覆盖`start_urls`中定义的URL，成为Spider的唯一初始请求。

4. **Engine将URL交给Downloader Middlewares处理，然后发送给Downloader下载：** 在URL发送给下载器之前，它会经过下载中间件的处理，可以对请求进行修改和预处理。然后，Engine将请求发送给下载器。

5. **Downloader接受服务器生成的Response：** 下载器负责与服务器进行通信，下载网页内容，并将服务器返回的响应封装为`scrapy.http.Response`对象。

6. **Engine将服务器生成的Response交给Spider Middlewares处理，然后发送给Spider处理：** 响应首先经过Spider Middlewares的处理，然后Engine将响应传递给Spider中的回调函数进行处理。

7. **Spider的工作是解析Response：** Spider的回调函数负责解析响应内容。如果内容是需要需要生成新的请求URL，以`yield scrapy.Request`的形式返回给Engine。如果内容是数据需要进一步处理，Engine将Item传递给Item Pipeline进行处理。

8. **重复第2步到第7步，直到Scheduler中没有更多的Request：** 引擎会继续重复这个过程，直到Scheduler中没有更多的请求要处理，然后引擎关闭该网站，爬取任务结束。

## 2.4 Scrapy文件

### 2.4.1 **spider_name.py**

```python
import scrapy


class SpiderNameSpider(scrapy.Spider):
    name = 'spider_name'
    allowed_domains = ['domain.com']
    start_urls = ['http://domain.com/']
	
    # 解析函数，parse特殊用处，针对start_urls
    def parse(self, response):
        pass
    # 自定义解析函数
    def parse_page2(self,response):
        pass
    
    # 自定义起始请求,
     # yield scrapy.Request(url, callback)加入调度队列，callback是解析函数
     def start_requests(self):
        urls1 = 'https://example1.com'
        urls2 = 'https://example2.com'
        yield scrapy.Request(url1, callback=self.parse)
        yield scrapy.Request(url2, callback=self.parse_page2)
        urls = ['https://example3.com', 'https://example4.com']
        for url in urls:
            yield scrapy.Request(url, callback=self.parse)
    
    def close(self, reason):
        # 在Spider关闭时执行一些清理操作
        self.logger.info(f'Spider closed, reason: {reason}')
        # 释放资源或执行其他操作
```

说明：

1. 初始URL来自start_urls或者start_requests函数，这是整个项目开始的地方

2. 如果没有start_requests函数，start_urls会自动加入调度队列，解析函数也是parse。

3. 如果有start_requests函数，会覆盖start_urls自动加入调度队列的过程。

4. 自定义加入调度队列的函数是scrapy.Request，其参数如下：

   **`scrapy.Request(url, callback=None, method='GET', headers=None, body=None, cookies=None, meta=None, encoding='utf-8', priority=0, dont_filter=False, errback=None, flags=None)`**

   - `url` (str): 要请求的URL地址。
   - `callback` (callable): 指定处理响应的**回调函数**。这个函数将在收到响应后被调用，通常用于**解析HTML内容和提取数据**。
   - `method` (str): HTTP请求方法，可以是'GET'、'POST'等。
   - `headers` (dict): 包含HTTP请求头的字典。可以用于设置请求的User-Agent、Referer、Cookie等信息。
   - `body` (str or bytes): HTTP请求的正文数据，通常用于POST请求。
   - `cookies` (dict or list): 一个字典或CookieJar对象，用于发送HTTP请求时包含的Cookie信息。
   - `meta` (dict): 一个包含任意元数据的字典，用于传递额外的信息给回调函数。
   - `encoding` (str): **指定响应内容的编码方式，通常是'utf-8'**。
   - `priority` (int): 请求的优先级，数字越大优先级越高。
   - `dont_filter` (bool): 如果为True，将防止Scrapy对此请求进行URL去重。默认为False。
   - `errback` (callable): 用于处理请求错误的回调函数。如果请求发生错误，将调用此函数处理。
   - `flags` (list): 一个包含请求标志的列表，用于控制请求的行为。

5. `closed()`，当 Spider 关闭时，该方法会被调用.在这里一般会定义释放资源的一些操作或其他收尾操作。

### 2.4.2 **middlewares.py**

1. **预设部分**

```python
from scrapy import signals

# useful for handling different item types with a single interface
from itemadapter import is_item, ItemAdapter

#预定义的爬虫中间件
class ProjectNameSpiderMiddleware:
    # Not all methods need to be defined. If a method is not defined,
    # scrapy acts as if the spider middleware does not modify the
    # passed objects.

    @classmethod
    def from_crawler(cls, crawler):
        # This method is used by Scrapy to create your spiders.
        s = cls()
        crawler.signals.connect(s.spider_opened, signal=signals.spider_opened)
        return s

    def process_spider_input(self, response, spider):
        # Called for each response that goes through the spider
        # middleware and into the spider.

        # Should return None or raise an exception.
        return None

    def process_spider_output(self, response, result, spider):
        # Called with the results returned from the Spider, after
        # it has processed the response.

        # Must return an iterable of Request, or item objects.
        for i in result:
            yield i

    def process_spider_exception(self, response, exception, spider):
        # Called when a spider or process_spider_input() method
        # (from other spider middleware) raises an exception.

        # Should return either None or an iterable of Request or item objects.
        pass

    def process_start_requests(self, start_requests, spider):
        # Called with the start requests of the spider, and works
        # similarly to the process_spider_output() method, except
        # that it doesn’t have a response associated.

        # Must return only requests (not items).
        for r in start_requests:
            yield r

    def spider_opened(self, spider):
        spider.logger.info('Spider opened: %s' % spider.name)

# 预定义的下载中间件
class ProjectNameDownloaderMiddleware:
    # Not all methods need to be defined. If a method is not defined,
    # scrapy acts as if the downloader middleware does not modify the
    # passed objects.

    @classmethod
    def from_crawler(cls, crawler):
        # This method is used by Scrapy to create your spiders.
        s = cls()
        crawler.signals.connect(s.spider_opened, signal=signals.spider_opened)
        return s

    def process_request(self, request, spider):
        # Called for each request that goes through the downloader
        # middleware.

        # Must either:
        # - return None: continue processing this request
        # - or return a Response object
        # - or return a Request object
        # - or raise IgnoreRequest: process_exception() methods of
        #   installed downloader middleware will be called
        return None

    def process_response(self, request, response, spider):
        # Called with the response returned from the downloader.

        # Must either;
        # - return a Response object
        # - return a Request object
        # - or raise IgnoreRequest
        return response

    def process_exception(self, request, exception, spider):
        # Called when a download handler or a process_request()
        # (from other downloader middleware) raises an exception.

        # Must either:
        # - return None: continue processing this exception
        # - return a Response object: stops process_exception() chain
        # - return a Request object: stops process_exception() chain
        pass

    def spider_opened(self, spider):
        spider.logger.info('Spider opened: %s' % spider.name)
```

2. 自定义部分

```python
# 下载器中间件
class MyDownloaderMiddleware1:
    def __init__(self, settings, user_agent):
        self.settings = settings
        self.user_agent = user_agent

    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        settings = crawler.settings
        user_agent = settings.get('USER_AGENT')
        return cls(settings, user_agent)

    def process_request(self, request, spider):
        # 处理请求的中间件逻辑

    def process_response(self, request, response, spider):
        # 处理响应的中间件逻辑

    def process_exception(self, request, exception, spider):
        # 处理异常的中间件逻辑

        
class MyDownloaderMiddleware2:
    pass


# 爬虫中间件
class MySpiderMiddleware1:
    def __init__(self, user_agent):
        self.user_agent = user_agent

    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        user_agent = crawler.settings.get('USER_AGENT')
        return cls(user_agent)

    def process_spider_input(self, response, spider):
        # 在Spider解析之前对响应进行处理的中间件逻辑
        response.headers['User-Agent'] = self.user_agent

    def process_spider_output(self, response, result, spider):
        # 处理Spider输出结果的中间件逻辑
        for item in result:
            yield item
            
    def process_spider_exception(self, response, exception, spider):
        # 处理Spider在处理请求时抛出的异常
        pass
    
    def process_start_requests(self, start_requests, spider):
        # 处理Spider的初始请求生成器
        for request in start_requests:
            yield request

    def spider_opened(self, spider):
        # 当爬虫被打开时执行的中间件逻辑
        spider.log(f'Spider {spider.name} opened')

    def spider_closed(self, spider):
        # 当爬虫被关闭时执行的中间件逻辑
        spider.log(f'Spider {spider.name} closed')

        
class MySpiderMiddleware2:
    pass
```

- **中间件说明**
  - 包括两种，**下载器中间件（Downloader Middleware）和爬虫中间件（Spider Middleware）**，分别处理**爬虫程序不同的时期**。
  - 本质是写在**middlewares.py的普通类**。
  - **类名没有要求，也无关是哪种类型的中间件。**
  - **在settings.py声明**，是下载器中间件或者爬虫中间件，以及**优先级**。

1. 下载器中间件（Downloader Middleware）
   - 参考：[下载器中间件](https://www.kingname.info/2018/11/18/know-middleware-of-scrapy)
   - 3个方法
     - process_request(request, spider)
     - process_response(request, response, spider)
     - process_exception(request, exception, spider)

### 2.4.x **settings.py**

`from scrapy.conf import settings`