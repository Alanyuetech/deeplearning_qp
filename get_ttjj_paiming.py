from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import pandas as pd
import time
from selenium.webdriver.common.by import By

url = "http://fund.eastmoney.com/data/fundranking.html#tall;c0;r;szzf;pn20000;ddesc;qsd20230305;qed20230504;qdii;zq;gg;gzbd;gzfs;bbzt;sfbb"

chrome_options = Options()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--disable-gpu')

driver = webdriver.Chrome(options=chrome_options)
driver.get(url)

data = []
columns = ['基金代码', '基金简称', '日期', '单位净值', '累计净值', '日增长率', '近1周', '近1月', '近3月', '近6月', '近1年', '近2年', '近3年', '近年来', '成立来', '自定义','手续费']


    
html = driver.page_source
soup = BeautifulSoup(html, 'html.parser')

table = soup.select_one("#dbtable")  # Find the table using CSS selector
rows = table.find_all('tr')

for row in rows[1:]:
    cols = row.find_all('td')
    cols = [col.text.strip() for col in cols]
    selected_cols = cols[2:19]  # Select only the required columns
    data.append(selected_cols)



driver.quit()

df = pd.DataFrame(data, columns=columns)
print(df)


