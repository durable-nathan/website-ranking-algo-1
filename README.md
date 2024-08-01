
## Configuration

Edit the file `html_path` inside build-sample.py to the website you want to scrape.

## Running the script

First, run the embedder. You can do this with `modal run build-sample.py`. Once that has been ran, you can run `python main.py` to see the results.

## What we can do with this
with this data, we can tell if the user has an image that can be improved/more closer to the content
we can train a model that would try to optimize this score
we now have a way of describing a "nice website" to the computer. This is the first step if we want to train our own models

## How it works
These are my notes on the first version of a function for scoring the layout and information relevance of a website section.
we can extract bounding boxes around elements of a section on the website.
L: to score the layout
we reconstruct a basic version of the section, and vertically cut the matrix in half
we flip the half, and compare its structure using the SSIM index
the intuition here is we want to reward layouts that are vertically symmetrical
I: to score the content information
we extract embeddings for images and text.
we calculate the cosine similarity of the embedding compared to all other documents, and we sum the result
the intuition here is to keep each website section relevant to a topic
C: to score the colour
tbd
maybe something like this - https://observablehq.com/@sebastien/a-quantitative-approach-to-colour-palette-selection#

**Formula**
website_design_score(bounding_boxes) = L * I * C

## Results
Example 1: bad website
Symmetry score: 0.5137456666463186
- symmetry score here is bad because the left and right side is not balanced
Content similarity score: 0.5676669551056587
- this is the sum of all the edges / amount of edges  
- I dont know why this is bad right now. gotta fix this...
- the lines and numbers you see that connect the blue squares is the the content similarity score. It is the embedding of the text/image * length of the distance.
- this is still a WIP: the similarity score for image and text is low when it should be higher... still trying to figure out why 
- my thoughts: is that the closer the content the more relevant it should be - we should prob include some sort of weight
<div style="display: flex; flex-wrap: wrap">
<img src="https://github.com/user-attachments/assets/332d4781-e4e3-4166-8e15-65144fa3c54e">
  <img src="https://github.com/user-attachments/assets/c88d5a5b-f00a-4726-b6e4-a9feb876d3ed">
  
</div>




Example 2: Good website
Symmetry score: 0.8017992337740113
- symmetry score here is good because the shape and area between left and right are somewhat close 
Content similarity score: 0.5686441161577919
- again, I dont know why this is bad :(
<img src="https://github.com/user-attachments/assets/0b6508de-80e3-4083-b523-c5356b0eae63">
<img src="https://github.com/user-attachments/assets/f6b4e9b5-bf04-4396-91b6-53191c687ee3">

## Next steps
Currently, the algorithm takes in an array of objects of bounding boxes. The next steps would be to:
- Build a scraper that would extract bounding boxes, text and image content from html
- Tweak the way the scores are combined - at the moment each score is independent of each other. It could also use some damping, etc 
- I think for layout:
  - we also need to incorporate some score that represents how closely it follows the design style guide
  - we also need to somehow represent element sizes as well (e.g. less important elements should be sized smaller)
- Complete the colour ranking score



