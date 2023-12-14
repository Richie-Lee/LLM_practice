from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time
from datetime import datetime

# Get HMTL source
SCRAPE_HTML_SOURCE = True
EXPORT = False

# Track total runtime
_start_time = datetime.now()
bdo_uk_url = "https://www.bdo.co.uk" # used for formulating urls from scraped documents

# URL to scrape
url = "https://www.bdo.co.uk/en-gb/insights/tax"
export_directory = "C:/Users/RLee/Desktop/TAX BASE/bdo_uk_scrape.txt"

num_clicks = 1

def scrape_html_source(url):
    """
    Scrapes the html file from the URL.
    - Capable of handling dynamic pages, by automating hovering
    """
    # Setup Chrome options
    chrome_options = Options()
    # chrome_options.add_argument("--headless")  # Uncomment to run headless
    chrome_options.add_argument("--no-sandbox")  # Bypass OS security model
    chrome_options.add_argument("--disable-gpu")  # Disable GPU hardware acceleration
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    # Initialize Chrome WebDriver
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    try:
        # Open the webpage
        driver.get(url)
    
        # Wait until the specific element that signifies that articles are loaded is present
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "insight-grid__cards"))
        )
        
        # Locate all article elements
        article_elements = WebDriverWait(driver, 10).until(
            EC.visibility_of_all_elements_located((By.CLASS_NAME, "animated-card"))
        )
    
        # Hover over each article element to trigger the loading of content
        for article in article_elements:
            # Scroll the article into view before hovering to avoid MoveTargetOutOfBoundsException
            driver.execute_script("arguments[0].scrollIntoView(true);", article)
            time.sleep(1)  # Wait for scrolling and possible page re-layout
            ActionChains(driver).move_to_element(article).perform()
            # You may need to wait for the animation to complete
            # time.sleep(1)  # Adjust the sleep time as necessary
    
        # Scroll a bit further down to ensure the button is fully in view
        driver.execute_script("window.scrollBy(0, 500);")  # Adjust the Y value as necessary
        time.sleep(3)  # Wait for scrolling and possible page re-layout
        
        for _ in range(12):  # Adjust the range for the number of times you need to click
            show_more_xpath = "//a[contains(@class, 'btn--secondary') and .//span[contains(text(), 'Show More')]]"
            show_more_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, show_more_xpath))
            )
                            
            # Attempt to click the "show more" button
            try:
                # Scroll the button into view
                driver.execute_script("arguments[0].scrollIntoView(true);", show_more_button)
                # Scroll up a bit if there is a fixed header
                driver.execute_script("window.scrollBy(0, -150);")
                # Wait for any possible overlays to disappear
                WebDriverWait(driver, 10).until(
                    EC.invisibility_of_element((By.CSS_SELECTOR, "div.overlay"))  # Replace with actual overlay selector
                )
                
                # Try clicking the inner span if the outer <a> tag is not working
                inner_span = show_more_button.find_element(By.XPATH, ".//span[contains(text(), 'Show More')]")
                driver.execute_script("arguments[0].click();", inner_span)
            
                # Wait for the new content to load
                time.sleep(3)  # Adjust this time as needed
                
            except Exception as e:
                print(f"Failed to click 'Show More' button: {e}")

    except Exception as e:
        print("error")
        # import traceback
        # print(e)
        # traceback.print_exc()  # This will print the full traceback
        
    finally:
        # Re-grab the page source after all hovers have been performed
        html_source = driver.page_source
        
        # Close driver
        driver.quit()
    
    return html_source

if SCRAPE_HTML_SOURCE == True:
    html_source = scrape_html_source(url)
    
soup = BeautifulSoup(html_source, 'lxml')

# Find the container for all articles
article_container = soup.find("div", class_="insight-grid__cards")
articles = article_container.find_all("article") 

export_str = ""
article_count = 0 # BDO UK insights (Tax) should be 255, up to january 2021

for article in articles:
    title = article.find("div", class_="featured-card__title").text if article.find("div", class_="featured-card__title") else None
    date = article.find("div", class_="featured-card__date").text if article.find("div", class_="featured-card__date") else None
    description = article.find("div", class_="featured-card__description").text.strip() if article.find("div", class_="featured-card__description") else None
    
    link_tag = article.find('a', class_='animated-card cursor-pointer')  # this should find the 'a' tag directly
    link = bdo_uk_url + link_tag['href'] if link_tag else None  # gets the href attribute if the 'a' tag is found
    
    article_summary = f"Index: {article_count} \nTitle: {title} \nDate: {date} \nDescription: {description} \nLink: {link}\n-\n" # use \n-\n as splitting criteria 
    # print(article_summary)
    export_str += article_summary
    article_count += 1


def str_to_txt_file(export_str):
    text_file = open(export_directory, 'w')
    text_file.write(export_str)
    text_file.close()

if EXPORT == True:
    str_to_txt_file(export_str)
    
# Print execution time
print(f"\n===============================\nTotal runtime:  {datetime.now() - _start_time}")

