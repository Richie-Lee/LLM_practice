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
EXPORT = True

# Track total runtime
_start_time = datetime.now()
bdo_uk_url = "https://www.bdo.com" # used for formulating urls from scraped documents

# URL to scrape
url = "https://www.bdo.com/tax-policy"
export_directory = "C:/Users/RLee/Desktop/TAX BASE/bdo_us_scrape.txt"

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
    
        # Scroll a bit further down to ensure the cards are fully in view
        driver.execute_script("window.scrollBy(0, 500);")  # Adjust the Y value as necessary
        time.sleep(3)  # Wait for scrolling and possible page re-layout
        
        # Wait until the specific element that signifies that articles are loaded is present
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "card-grid"))
        )
        
        # Locate all article elements
        article_elements = WebDriverWait(driver, 10).until(
            EC.visibility_of_all_elements_located((By.CLASS_NAME, "content-wrapper"))
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
        
        for _ in range(55):  # Adjust the range for the number of times you need to click
            show_more_xpath = "//button[contains(@class, 'btn-secondary') and .//span[contains(text(), 'SHOW MORE')]]"
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
                inner_span = show_more_button.find_element(By.XPATH, ".//span[contains(text(), 'SHOW MORE')]")
                driver.execute_script("arguments[0].click();", inner_span)
            
                # Wait for the new content to load
                time.sleep(3)  # Adjust this time as needed
                
            except Exception as e:
                print(f"Failed to click 'SHOW MORE' button: {e}")

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
article_container = soup.find("div", class_="card-grid")
articles = article_container.find_all("div", class_="InsightCardWrapperStyled-sc-1w8ojf6-0 lcuLwr insight-card-wrapper") 

export_str = ""
article_count = 0 # BDO UK insights (Tax) should be 255, up to january 2021

for article in articles:
    title = article.find("h3").text if article.find("h3") else None
    date = article.find("span", class_="publish-date").text if article.find("span", class_="publish-date") else None
    description = article.find("p", class_="description").text.strip() if article.find("p", class_="description") else None
    
    link_tag = article.find('a', class_='no-underline-link')  # this should find the 'a' tag directly
    link = bdo_uk_url + link_tag['href'] if link_tag else None  # gets the href attribute if the 'a' tag is found
    article_summary = f"Index: {article_count} \nTitle: {title} \nDate: {date} \nDescription: {description} \nLink: {link}\n-\n" # use \n-\n as splitting criteria 
    print(article_summary)
    export_str += article_summary
    article_count += 1


def str_to_txt_file(export_str, export_directory):
    with open(export_directory, 'w', encoding='utf-8') as text_file:
        text_file.write(export_str)

if EXPORT == True:
    str_to_txt_file(export_str, export_directory)
    
# Print execution time
print(f"\n===============================\nTotal runtime:  {datetime.now() - _start_time}")

