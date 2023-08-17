#输入 date_set和n   date_set是一个日期字符串，n是1～7的整数。输出date_set前一周的星期n的日期
def get_lastweek_n(date_set,n):
    import datetime
    date_set = datetime.datetime.strptime(date_set,'%Y%m%d')
    date_set = date_set-datetime.timedelta(days=7)
    day_n = date_set.weekday()
    date_set = date_set-datetime.timedelta(days=day_n)+datetime.timedelta(days=n-1)
    return date_set.strftime('%Y%m%d')
print(get_lastweek_n('20230625',4))