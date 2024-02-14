from bs4 import BeautifulSoup
import requests
import datetime
import csv
import smtplib
import time

def check_price_ebay():
    URL = 'https://www.ebay.com/itm/1234567890'  # Replace with an actual eBay product URL
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.108 Safari/537.36",
        "Accept-Encoding": "gzip, deflate",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "DNT": "1",
        "Connection": "close",
        "Upgrade-Insecure-Requests": "1"
    }

    page = requests.get(URL, headers=headers)
    soup = BeautifulSoup(page.content, "html.parser")

    # Update these selectors based on your eBay listing
    title = soup.find("h1", {"id": "itemTitle"}).text.replace('Details about  \xa0', '')
    price = soup.find("span", {"id": "prcIsum"}).text.strip().replace('US $', '')
    
    # Clean up data
    title = title.strip()
    price = float(price.replace(',', ''))  # Convert price to float for comparison

    # Check if price is below a certain level and send email
    if price < 15:  # Set your target price here
        send_mail(title, URL, price)

    today = datetime.date.today()
    header = ['Title', 'Price', 'Date']
    data = [title, price, today]

    with open('eBayWebScraperDataset.csv', 'a+', newline='', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(data)

def send_mail(title, url, price):
    server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
    server.ehlo()
    server.login('your_email@example.com', 'your_password')
    
    subject = f"Price drop alert for {title}!"
    body = f"The price for {title} has dropped below your target to ${price}! Check it out here: {url}"
    
    msg = f"Subject: {subject}\n\n{body}"
    
    server.sendmail(
        'from_email@example.com',
        'to_email@example.com',
        msg
    )
    print("Email has been sent")
    server.quit()

# Example usage
while True:
    check_price_ebay()
    time.sleep(86400)  # Adjust frequency as needed
