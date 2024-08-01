from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from PIL import Image, ImageDraw
import time
import os
from collections import defaultdict

from sklearn.cluster import DBSCAN

def get_height(driver):
    # return driver.execute_script("return document.body.scrollHeight")
    # Find the last element in the document
    last_element = driver.find_element(By.XPATH, '//*')
    
    # Get the height of the last element to determine the total height
    total_height = last_element.location['y'] + last_element.size['height']
    return total_height


def calculate_center(bbox):
    x = bbox["x"]
    y = bbox["y"]
    w = bbox["width"]
    h = bbox["height"]

    cx = x + w / 2
    cy = y + h / 2
    return cx, cy

def extract_bounding_boxes_by_section(driver):
    sections = driver.find_elements(By.CSS_SELECTOR, 'section')
    sections_bboxes = {}
    
    for section_index, section in enumerate(sections):
        section_rectangles = []

        time.sleep(2)
        elements = section.find_elements(By.CSS_SELECTOR, 'h1, h2, h3, h4, h5, h6, p, button, a, img')

        # X should be relative to the section
        section_x = section.location['x']
        section_y = section.location['y']

        for element in elements:
            rect = {
                'x': element.location['x'] - section_x,
                'y': element.location['y'] - section_y,
                'width': element.size['width'],
                'height': element.size['height'],
                'type': element.tag_name,
            }
            x, y = calculate_center(rect)
            rect['center'] = [x, y]

            if element.tag_name == 'img':
                rect['src'] = element.get_attribute('src')
                rect['alt'] = element.get_attribute('alt')
            else:
                rect['content'] = element.text

            section_rectangles.append(rect)

        sections_bboxes[f'section_{section_index}'] = section_rectangles
        



    return sections_bboxes

def cluster_bounding_boxes(rectangles):
    """
    CLUSTER BOUNDING BOXES
    """
    # Extract centers of bounding boxes
    centers = [rect['center'] for rect in rectangles]

    # Cluster centers
    clustering = DBSCAN(eps=300, min_samples=2).fit(centers)

    # Get cluster labels
    labels = clustering.labels_

    # Assign cluster labels to rectangles
    for index, rect in enumerate(rectangles):
        rect['cluster'] = labels[index]

    # Group rectangles by cluster label
    clusters = defaultdict(list)
    for rect in rectangles:
        clusters[rect['cluster']].append(rect)
    
    # Convert clusters to a list of lists
    clusters_list = list(clusters.values())
    
    return clusters_list

def build_screenshot(driver):
    """
    Build screen shot
    """
    # Get the total height of the page
    total_height = get_height(driver)
    print("Total height:", total_height)
    current_scroll_position = 0
    scroll_increment = 1024
    screenshot_paths = []

    while current_scroll_position < total_height:
        # Scroll the window
        driver.execute_script(f"window.scrollTo(0, {current_scroll_position});")
        time.sleep(2)  # Allow time for the page to render after scrolling

        # Take a screenshot of the current view
        screenshot_path = f'images/screenshot_{current_scroll_position}.png'
        driver.save_screenshot(screenshot_path)
        screenshot_paths.append(screenshot_path)

        current_scroll_position += scroll_increment

    # Merge screenshots
    images = [Image.open(path) for path in screenshot_paths]
    total_width = images[0].width
    total_height = sum(image.height for image in images)

    merged_image = Image.new('RGB', (total_width, total_height))
    y_offset = 0

    for image in images:
        merged_image.paste(image, (0, y_offset))
        y_offset += image.height

    # Scale the merged image to width 1400 while maintaining aspect ratio
    new_width = 1400
    aspect_ratio = merged_image.height / merged_image.width
    new_height = int(new_width * aspect_ratio)
    merged_image = merged_image.resize((new_width, new_height))

    # Clean up individual screenshots
    for path in screenshot_paths:
        os.remove(path)

    return merged_image

def scrape(url):
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

    # URL to your web page
    driver.get(url)

    # Allow some time for the page to load
    time.sleep(2)

    print("Extracting bounding boxes")
    merged_image = build_screenshot(driver)
    sections_bboxes = extract_bounding_boxes_by_section(driver)
    #clusters = cluster_bounding_boxes(rectangles)
    print("Done extracting bounding boxes")

    driver.quit()

    # Draw bounding boxes on the scaled image
    draw = ImageDraw.Draw(merged_image)

    # Draw rectangles onto image
    print("Drawing bounding boxes")
    for section_id, rectangles in sections_bboxes.items():
        for index, rect in enumerate(rectangles):
            scaled_rect = {
                'x': rect['x'],
                'y': rect['y'],
                'width': rect['width'],
                'height': rect['height']
            }
            draw.rectangle([(scaled_rect['x'], scaled_rect['y']), (scaled_rect['x'] + scaled_rect['width'], scaled_rect['y'] + scaled_rect['height'])], outline="red", width=2)

    # Draw cluster rectangles
    """
    print("Drawing cluster rectangles", len(clusters))
    for cluster in clusters:
        # Compute the bounding box of the cluster
        if not cluster:
            continue
        min_x = min(rect['x'] for rect in cluster)
        min_y = min(rect['y'] for rect in cluster)
        max_x = max(rect['x'] + rect['width'] for rect in cluster)
        max_y = max(rect['y'] + rect['height'] for rect in cluster)

        # Draw the rectangle around the cluster
        draw.rectangle([min_x, min_y, max_x, max_y], outline="green", width=2)
    """


    # Save the scaled image with bounding boxes
    output_path = 'images/output.png'
    merged_image.save(output_path)

    print(f"Bounding boxes drawn and saved to {output_path}")

    return sections_bboxes

#r = scrape("https://testdurable2.durablesites.com/?pt=NjY2MzdkMmRmZDE3MzkxOGU3MzE3MTFlOjE3MjI0NTU2NDMuNjU5OnByZXZpZXc=")
#print(r)
