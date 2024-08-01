import numpy as np
#from skimage.metrics import structural_similarity as ssim
#from scrapers.local import scrape
from scrapers.public_section import scrape
from durable_sections import sections as rectangles, plot_bounding_boxes
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

from ssim import ssim

#html_path = 'file:///Users/nathanlu/code/durable/lab/ranking/_test_good.html'
html_path = 'file:///Users/nathanlu/code/durable/lab/ranking/_test_full.html'
rectangles1 = scrape(html_path)

# Encode each tag as color, that way
# we can include this information in the SSIM
html_tags_encoding = {
    ("h1", "h2", "h3", "h4", "h5", "h6", "p"): (255, 0, 0),
    ("img"): (0, 255, 0),
    ("a", "button"): (0, 0, 255),
}

durable_sections_img_map = {
    0: "section_0.jpeg",
    1: "section_1.jpeg",
    2: "section_2.jpeg",
    3: "section_3.png",
    4: "section_4.png",
}


def create_matrix_for_bounding_box(bboxes):
    width = 1440
    height = 1024 

    # Create a matrix of zeros
    matrix = np.zeros((height, width, 3), dtype=np.uint8)

    # Fill the matrix with the bounding boxes
    for bbox in bboxes:
        x = bbox["x"]
        y = bbox["y"]
        w = bbox["width"]
        h = bbox["height"]

        # Get the color for the tag
        tag = bbox["type"]
        color = None
        for tags, c in html_tags_encoding.items():
            if tag in tags:
                color = c
                break

        # set the color, ( 255,0,0)
        matrix[y:y+h, x:x+w] = color

    # return gray scale image
    matrix = cv2.cvtColor(matrix, cv2.COLOR_BGR2GRAY)

    return matrix

def compare_structural_sim(bb1, bb2):
    #mat1 = create_matrix_for_bounding_box(bb1)

    rectangles1 = list(bb1.values())
    mat1 = [create_matrix_for_bounding_box(bb) for bb in rectangles1]

    rectangles2 = list(bb2.values())
    #rectangles = [item for sublist in bb2.values()]
    durable_sections_matrices = [create_matrix_for_bounding_box(bb) for bb in rectangles2]

    final_image = []

    # compare ssim for each durable section
    for index, section in enumerate(mat1):
        ssim_scores = [calculate_symmetry_score(section, mat2) for mat2 in durable_sections_matrices]
        # find highest ssim score and its index
        max_ssim_score = max(ssim_scores)
        max_ssim_index = ssim_scores.index(max_ssim_score)
        print(f"section {index} should be durable section {max_ssim_index}. scored at {max_ssim_score}")
        final_image.append(max_ssim_index)


    # load sections from durable_sections_img_map and build the final image
    final_image = [cv2.imread(f"images/{durable_sections_img_map[index]}") for index in final_image]
    # save


    """
    SAVE FINAL IMAGE
    """
    target_width = 3456

    # Pad each image to the target width
    adjusted_images = []
    for img in final_image:
        if img.shape[1] < target_width:
            # Pad the image
            pad_width = target_width - img.shape[1]
            padded_img = np.pad(img, ((0, 0), (0, pad_width), (0, 0)), mode='constant', constant_values=0)
            adjusted_images.append(padded_img)
        else:
            # No adjustment needed
            adjusted_images.append(img)
    
    # Now stack the adjusted images
    final_stacked_image = np.vstack(adjusted_images)
    cv2.imwrite("final_image.png", final_stacked_image)
    

    # Save the matrices for the durable sections
    for i, section in enumerate(mat1):
        cv2.imwrite(f'images/original_{i}.png', section)
    for i, mat in enumerate(durable_sections_matrices):
        print(f"score for {i}", ssim_scores[i])
        cv2.imwrite(f'images/durable_{i}.png', mat)



def calculate_symmetry_score(x, y):
    #ssim_index, _ = ssim(x, y, data_range=1.0, full=True)

    ssim_value = ssim(x, y)
    print("got ssim value", ssim_value)
    return ssim_value



compare_structural_sim(rectangles1, rectangles)
plot_bounding_boxes(rectangles)
