from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import pandas as pd

url = 'https://www.amazon.in/gp/bestsellers/books/'
request = Request(url, headers={'User-agent': 'Mozilla/5.0'})
html = urlopen(request)
soup = BeautifulSoup(html, 'html.parser')
books = soup.find_all('div', id="gridItemRoot")

ranks=[]
authors=[]
titles=[]
types=[]
reviewers=[]
prices=[]
ratings=[]

for book in books:
    rank = book.find('span', "zg-bdg-text").text.replace("#", "")
    ranks.append(rank)

    title = book.find('div',"_cDEzb_p13n-sc-css-line-clamp-1_1Fn1y").text
    titles.append(title)

    author = book.find('div', "a-row a-size-small").text
    authors.append(author)

    type = book.find('span', "a-size-small a-color-secondary a-text-normal").text
    types.append(type)

    reviewer = book.find('div', "a-icon-row").find('span', "a-size-small").text
    reviewers.append(reviewer)

    price = book.find('div', "_cDEzb_p13n-sc-price-animation-wrapper_3PzN2").find('div', "a-row")\
        .find('span', "a-size-base a-color-price").find('span', "_cDEzb_p13n-sc-price_3mJ9Z").text.replace( "₹"," " )
    prices.append(price)

    rating = book.find('div', "a-icon-row").find('a', "a-link-normal").text
    ratings.append(rating[0:3])

pd.DataFrame({
    'Rank': ranks,
    'Title': titles,
    'Author': authors,
    'Rating (Out of 5)': ratings,
    'Price (INR)': prices,
    'Cover Type': types,
    'No of reviews': reviewers,
}).to_csv('best_seller.csv', index=False)
