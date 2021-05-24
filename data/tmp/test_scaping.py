import pandas as pd
from urllib.request import urlopen
from bs4 import BeautifulSoup

scraped = []
for page in range(1, 11):
    url = f'https://quotes.toscrape.com/page/{page}/'
    page = urlopen(url)
    soup = BeautifulSoup(page, 'html.parser')
    quotes = soup.find_all('div', class_='quote')

    for quote in quotes:
        text = quote.find('span', class_='text').text
        author = quote.find('small', class_='author').text
        tags = quote.find('div', class_='tags').find_all('a')

        tags_list = []
        for tag in tags:
            tags_list.append(tag.text)

        tags_str = str(tags_list)[1:-1]
        print(f'Text: {text}\nAuthor: {author}\nTags: {tags_str}\n\n')
        single_quote = [text, author, tags_list]
        scraped.append(single_quote)

df = pd.DataFrame(scraped, columns=['quote', 'author', 'tags'])
df.to_csv('quotes.csv', index=False)