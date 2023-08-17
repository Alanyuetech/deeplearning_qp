#写一个爬虫，爬取网站上的数据
import requests
from bs4 import BeautifulSoup
import re


# 爬虫主要框架
# 1. 定义一个函数，用来获取网页源代码
def get_html(url):
    try:
        r = requests.get(url)
        r.raise_for_status()
        r.encoding = r.apparent_encoding
        return r.text
    except:
        return ""
# 2. 定义一个函数，用来解析网页源代码
def parse_html(html):
    # 创建一个正则表达式对象，用来解析网页源代码
    pattern = re.compile('<.*?>')
    # 将网页源代码转换成BeautifulSoup对象
    soup = BeautifulSoup(html, 'html.parser')
    # 获取所有的标签
    tags = soup.find_all(pattern)
    # 遍历所有的标签
    for tag in tags:
        # 获取标签的内容
        content = tag.get_text()
        # 去除标签的内容
        content = content.strip()
        # 去除标签的内容
        content = content.replace('\n', '')
        # 去除标签的内容
        content = content.replace('\t', '')
        # 去除标签的内容
        content = content.replace(' ', '')
        # 去除标签的内容
        content = content.replace('\xa0', '')
        # 去除标签的内容
        content = content.replace('\u200b', '')
    # 返回解析好的网页源代码   
    return content



# 3. 定义一个函数，用来获取需要的数据
def get_data(soup):
    # 获取标题
    title = soup.select('.news-title')[0].get_text()
    # 获取时间
    time = soup.select('.time')[0].get_text()
    # 获取内容
    content = soup.select('.news-content')[0].get_text()
    return title, time, content
# 4. 通过网站的url获取网页源代码
def get_one_page(url):
    try:
        html = get_html(url)
        if html:
            soup = parse_html(html)
            title, time, content = get_data(soup)
            return title, time, content
        else:
            return "", "", ""
    except:
        return "", "", ""
url = 'http://fund.eastmoney.com/data/fundranking.html#tall;c0;r;s1nzf;pn50;ddesc;qsd20220410;qed20230410;qdii;zq;gg;gzbd;gzfs;bbzt;sfbb'
html = get_html(url)
soup = parse_html(html)
data = get_data(soup)