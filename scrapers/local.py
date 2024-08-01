from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from PIL import Image, ImageDraw
import time

def scrape(html_path):

    # Set up the Chrome options
    options = Options()
    options.add_argument('--headless')  # Run Chrome in headless mode
    options.add_argument('--disable-gpu')
    options.add_argument(f'--window-size=1400,1024')

    # Path to your ChromeDriver
    chrome_driver_path = '/usr/local/bin/chromedriver'

    # Initialize the WebDriver
    service = Service(chrome_driver_path)
    driver = webdriver.Chrome(service=service, options=options)
    driver.set_window_size(1400, 1024)

    # URL or path to your HTML file
    driver.get(html_path)

    # Allow some time for the page to load
    time.sleep(2)

    # Take a screenshot of the rendered page
    screenshot_path = 'images/original.png'
    # save screenshot at 1400, 1024
    driver.save_screenshot(screenshot_path)

    # Get the dimensions and positions of the elements
    elements = driver.find_elements(By.CSS_SELECTOR, 'h1, h2, h3, h4, h5, h6, p, button, a, img')
    rectangles = []

    for element in elements:
        rect = {
            'x': element.location['x'],
            'y': element.location['y'],
            'width': element.size['width'],
            'height': element.size['height'],
            'type': element.tag_name
        }
        if element.tag_name == 'img':
            rect['src'] = element.get_attribute('src')
            rect['alt'] = element.get_attribute('alt')
        else:
            rect['content'] = element.text

        rectangles.append(rect)

    driver.quit()

    # Print the rectangles
    print(rectangles)

    # Open the screenshot with Pillow
    image = Image.open(screenshot_path)
    new_width = 1400

    # Calculate the new height while maintaining the aspect ratio
    aspect_ratio = image.height / image.width
    new_height = int(new_width * aspect_ratio)

    # Resize the image
    image = image.resize((new_width, new_height))

    draw = ImageDraw.Draw(image)

    # Draw bounding boxes around the elements
    for rect in rectangles:
        draw.rectangle([(rect['x'], rect['y']), (rect['x'] + rect['width'], rect['y'] + rect['height'])], outline="red", width=2)

    # Save the image with bounding boxes
    output_path = 'images/output.png'
    image.save(output_path)

    print(f"Bounding boxes drawn and saved to {output_path}")

    return rectangles
