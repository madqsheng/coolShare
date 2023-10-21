import asyncio
import time

'''
修饰器给函数增加功能：计算函数消耗的时间
包括同步和协程
'''

#同步修饰器
def sync_timing_decorator(func):
    async def wrapper(*args,**kwargs):
        start_time = time.time()
        result = await func(*args,**kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"{func.__name__} 执行时间: {execution_time:.4f} 秒")
        return result
    return wrapper

#异步修饰器
def asyncio_timing_decorator(func):
    async def wrapper(*args,**kwargs):
        start_time = time.time()
        result = await func(*args,**kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"{func.__name__} 执行时间: {execution_time:.4f} 秒")
        return result
    return wrapper

#通用修饰器函数
def timing_decorator_err(func):
    async def async_wrapper(*args, **kwargs):
        if asyncio.iscoroutinefunction(func):
            start_time = time.time()
            result = await func(*args, **kwargs)
            end_time = time.time()
        else:
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
        execution_time = end_time - start_time
        print(f"{func.__name__} 执行时间: {execution_time:.4f} 秒")
        return result

#通用修饰器函数
def timing_decorator(func):
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"{func.__name__} 执行时间: {execution_time:.4f} 秒")
        return result

    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"{func.__name__} 执行时间: {execution_time:.4f} 秒")
        return result

    def wrapper(*args, **kwargs):
        if asyncio.iscoroutinefunction(func):
            return async_wrapper(*args, **kwargs)
        else:
            return sync_wrapper(*args, **kwargs)

    return wrapper

async def crawl_page(url):
    print('crawling {}'.format(url))
    sleep_time = int(url.split('_')[-1])
    await asyncio.sleep(sleep_time)
    print('OK {}'.format(url))

@timing_decorator
async def main(urls):
    print(urls)
    for url in urls:
        await crawl_page(url)

asyncio.run(main(['url_1', 'url_2', 'url_3', 'url_4']))
