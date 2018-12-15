import re
import pandas as pd
from bs4 import BeautifulSoup
import requests
import numpy as np
from urllib.parse import quote
import pickle
import random
from __future__ import print_function
from newspaper import Article
import csv

## 1-1. article crawling: day
def get_article(date):
    total = []

    pages = ['1']
    for page in pages:
        session = requests.Session()
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_5) AppleWebKit537.36 (KHTML, like Gecko) Chrome",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"}
        page_url = 'http://news.naver.com/main/list.nhn?mode=LS2D&mid=shm&sid2=264&sid1=100&date={}&page={}'.format(
            date, page)
        r = session.get(page_url, headers=headers)
        soup = BeautifulSoup(r.text, 'html.parser')
        dl_list = soup.find('div', {'class': 'list_body newsflash_body'}).findAll('dl')

        i = 0
        while i < len(dl_list):
            dl = dl_list[i]
            article_url = dl.findAll('dt')[-1].find('a')['href']
            article_cont = Article(article_url, language='ko')
            article_cont.download()
            try:
                article_cont.parse()
                article_title = article_cont.title
                article_cont = article_cont.text.replace("\t", "").replace("\n", "").replace("\r", "")
                total.append([article_title, article_cont])

            except:
                i -= 1
            i += 1

        if page == '1':
            try:
                page_range = soup.find('div', {'class': 'paging'}).findAll('a')
                page_range = [t.text for t in page_range if t.text != '다음']
                pages += page_range

            except:
                pages = pages

    return pd.DataFrame(total, columns=['article_title', 'article_cont'])


get_article('20180105')

## 1-2. article crawling: month

def get_all_article(date_list):
    total = pd.DataFrame()

    for date in date_list:
        date_article = get_article(date)
        total = pd.concat([total, date_article], ignore_index=True)

    return total


def make_date_list(month):
    month_28 = ['02']
    month_30 = ['04', '06', '09', '11']
    month_31 = ['01', '03', '05', '07', '08', '10', '12']

    date_range = ['0' + str(i) for i in range(1, 10)] + [str(i) for i in range(10, 32)]

    if month in month_28:
        date_list = ['2017' + month + t for t in date_range[:28]]
    elif month in month_30:
        date_list = ['2017' + month + t for t in date_range[:30]]
    elif month in month_31:
        date_list = ['2017' + month + t for t in date_range[:31]]

    return date_list


article_201703 = get_all_article(make_date_list('03'))

f = open('article_201703.txt', 'wb')
pickle.dump(article_201703, f)
f.close()

a = open('article_201703.txt', "rb")
pickle.load(a)

## 1-3. article crawling: year

def get_year_article(month_list):
    article_2017 = pd.DataFrame()
    for month in month_list:
        month_article = pd.read_pickle('article_2017' + month + '.txt')
        article_2017 = pd.concat([article_2017, month_article], ignore_index=True)

    return article_2017


month_list = ['0' + str(i) for i in range(1, 10)] + [str(i) for i in range(10, 13)]
article_2017 = get_year_article(month_list)

# error 제거
error_date = ['2017.12.23. 11:30 ~ 14:30', '2017.12.24. 05:30 ~ 08:30', '2017.12.23. 17:30 ~ 20:30',
              '2017.12.23. 20:30 ~ 23:30', '2017.12.24. 08:30 ~ 11:30']
error = [
    '8765432187654321876543218765432187654321876543218765432187654321스포츠연예IT/과학세계생활/문화사회경제정치다음이전검색어 입력||검색어 입력핫이슈뉴스토픽추출이 일시적으로 늦어져 최신 정보를 제공할 수 없습니다. {} 기준도움말'.format(
        t) for t in error_date]
error_location = [t not in error for t in article_2017.article_cont]


article_2017 = article_2017[error_location]
article_2017.index = np.arange(31513)


f = open('article_2017.txt', 'wb')
pickle.dump(article_2017, f)
f.close()

a = open('article_2017.txt', "rb")
article_2017 = pickle.load(a)


article_2017.to_csv('article_2017.csv', index=False)

article_2017 = pd.read_csv('article_2017.csv', encoding='cp949')


# 2. extract summarization: lexrank&textrank

from lexrankr import LexRank
from gensim.summarization import summarize

lexrank = LexRank()

def get_summarization(article_df, line_num, word_count):
    total_sum = []
    sum_type = []
    for i in range(len(article_df)):
        if i % 500 == 0:
            print(i)

        article = article_df.article_cont[i]
        if article[0] == '[':
            try:
                finish = article.index(']') + 1
                remove_str = article[:finish]
                article = article.replace(remove_str, "")

            except:
                article = article

        try:
            lexrank.summarize(article)
            summarization = lexrank.probe(line_num)  # lexrank
            summarization = summarization[0] + " " + summarization[1]
            sum_textrank = summarize(article, word_count=word_count)  # textrank

            if len(sum_textrank) > 0 and len(summarization) > len(sum_textrank):
                summarization = sum_textrank
                sum_type.append("textrank")
            else:
                sum_type.append("lexrank")

            summarization = re.sub('[-=#\/:$()\[\]\"}]', '', summarization)
            summarization = re.sub("[\'!?\.\,·◇△]", '', summarization)
            summarization = summarization.replace("\n", "").replace("기자", "")

            total_sum.append(summarization)

        except:
            summarization = article
            summarization = re.sub('[-=#/:$()\\[\\]\\"}]', '', summarization)
            summarization = re.sub("[\'!?\.\,·◇△]", '', summarization)
            summarization = summarization.replace("\n", "").replace("기자", "")

            total_sum.append(summarization)
            sum_type.append("article_cont")

    total_sum = pd.DataFrame(total_sum, columns=['summarization'])
    sum_type = pd.DataFrame(sum_type, columns=['sum_type'])

    return pd.concat([article_df, total_sum, sum_type], axis=1)


summarization_df = get_summarization(article_2017, 2, 50)

sum(list(summarization_df.sum_type != "article_cont"))

f = open('article_final_data.txt', 'wb')
pickle.dump(summarization_df, f)
f.close()

a = open('article_final_data.txt', "rb")
sum_df = pickle.load(a)

summarization_df.to_csv("article_final_data.csv", index=False)

pd.read_csv("article_final_data.csv", encoding="cp949")

# 3. POS tagging: NN(명), NR(수), VV(동), VA(형)

import pickle
import pandas as pd

from konlpy.tag import Kkma

kkma = Kkma()

a = open('article_only_summarization.txt', 'rb')
sum_df = pickle.load(a)


def make_pos(sum_df):
    pos_titles = pd.DataFrame([0] * len(sum_df), columns=['pos_title'])
    pos_articles = pd.DataFrame([0] * len(sum_df), columns=['pos_article'])
    pos_sums = pd.DataFrame([0] * len(sum_df), columns=['pos_sum'])

    for i in range(len(sum_df)):
        pos_title = kkma.pos(sum_df.article_title[i])
        pos_article = kkma.pos(sum_df.article_cont[i])
        pos_sum = kkma.pos(sum_df.summarization[i])

        pos_title = [t for t in pos_title if t[1][0:2] in ["NN", "NR", "VV", "VA"]]
        pos_article = [t for t in pos_article if t[1][0:2] in ["NN", "NR", "VV", "VA"]]
        pos_sum = [t for t in pos_sum if t[1][0:2] in ["NN", "NR", "VV", "VA"]]

        pos_titles.pos_title[i] = pos_title
        pos_articles.pos_article[i] = pos_article
        pos_sums.pos_sum[i] = pos_sum

    return pd.concat([pos_titles, pos_articles, pos_sums], axis=1)


sum_pos = make_pos(sum_df)

kk = []
for kkk in sum_pos.pos_sum:
    kk = kk + kkk
len(set(kk))

f = open('summarization_pos.txt', 'wb')
pickle.dump(sum_pos, f)
f.close()

a = open('summarization_pos.txt', 'rb')
sum_pos = pickle.load(a)


def make_sum_input(sum_pos):
    input_sums = pd.DataFrame([0] * len(sum_pos), columns=['pos_sum'])
    input_articles = pd.DataFrame([0] * len(sum_pos), columns=['pos_article'])

    for i in range(len(sum_pos)):
        pos_sum = ""
        for tag in sum_pos.pos_sum[i]:
            pos_sum = pos_sum + " " + tag[0]

        pos_article = ""
        for tag in sum_pos.pos_article[i]:
            pos_article = pos_article + " " + tag[0]

        input_sums.pos_sum[i] = pos_sum
        input_articles.pos_article[i] = pos_article

    return pd.concat([input_sums, input_articles], axis=1)


sum_input = make_sum_input(sum_pos)

a = open('title_pos.txt', 'rb')
sum_title = pickle.load(a)

yg_input = pd.concat([sum_input.pos_sum, sum_title.pos_title], axis=1)

sb_input = pd.concat([sum_title.pos_title, sum_title.pos_title_origin], axis=1)

f = open('yg_input.txt', 'wb')
pickle.dump(yg_input, f)
f.close()

f = open('sb_input.txt', 'wb')
pickle.dump(sb_input, f)
f.close()

a = open('yg_input.txt', 'rb')
pickle.load(a)

a = open('sb_input.txt', 'rb')
pickle.load(a)

yg_input.to_csv('yg_input.csv', index=False)

sb_input.to_csv('sb_input.csv', index=False)

pd.read_csv('yg_input.csv', encoding="cp949")

pd.read_csv('sb_input.csv', encoding="cp949")

# 4. word2vec

a = open('yg_input.txt', 'rb')
sum_pos = pickle.load(a)

len(list(sum_pos.pos_sum) + list(sum_pos.pos_title))

words = []
for word in list(sum_pos.pos_sum) + list(sum_pos.pos_title):
    word = word.split()
    word = [t for t in word if len(t) > 0]
    words.append(word)

# word2vec model for train set
from gensim.models import Word2Vec

model = Word2Vec(words, size=300, min_count=3, workers=4, sg=1)

# word2vec embedding vectors for words
w2v_sg = dict(zip(model.wv.index2word, model.wv.syn0))

f = open('w2v_pos_sum.txt', 'wb')
pickle.dump(w2v_sg, f)
f.close()

a = open('w2v_pos_sum.txt', 'rb')
pickle.load(a)