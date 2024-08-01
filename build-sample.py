from embedder import get_one_embedding
#from openai.embeddings_utils import cosine_similarity
from lib.modal import app
from scrapers.public import scrape
from openai import OpenAI 
from scipy import spatial
import numpy as np

client = OpenAI(api_key="")

txt_sim = "text-similarity-davinci-001"

def get_embedding(text, model="text-embedding-3-large"):
    #model = txt_sim
    
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=model).data[0].embedding

# Function to compute cosine similarity
def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dot_product = np.dot(vec1, vec2)
    print("got dot", dot_product)


    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

def label(imageUrl):
    completion = client.chat.completions.create(
      model="gpt-4o",
      messages=[
        {
          "role": "user",
          "content": [
            {"type": "text", "text": "Generate me a short description for this image. Be as concise as possible. Only respond with the label. This is for the alt of an image"},
            {
              "type": "image_url",
              "image_url": {
                "url": imageUrl,
              },
            },
          ],
        }
      ],
    )
    output = completion.choices[0].message.content
    print("got output", output)
    return output



"""
async def run(i):
    print("Starting generation for:", i)
    output = output.replace("```html", "")
    output = output.replace("```", "")
    print("Got generation for:", i)

    take_screenshot(output, f"output_{i}.png", "screenshots")
    #output_json = json_repair.loads(output)
    return output
"""



"""
rectangles = [                 
    {
        'x': 122,
        'y': 248,
        'width': 588,
        'height': 588,
        'type': 'img',
        'src': 'https://cdn.prod.website-files.com/632df91dd7c99c0ac992c47b/650da8b9358d0e23d1249bc6_Browser-hero.png',
    },
    {
        'x': 756,
        'y': 248,
        'width': 562,
        'height': 196,
        'type': 'h1',
        'content': 'AI that builds a website for you.',
    },
    {
        'x': 756,
        'y': 493,
        'width': 562,
        'height': 211,
        'type': 'p',
        'content': 'Get your business online in 30 seconds with the #1 AI website builder and marketing platform.',
    },
    {
        'x': 756,
        'y': 724,
        'width': 213,
        'height': 54,
        'type': 'button',
        'content': 'Start now',
    },
]
"""

"""
def main():

    # get embeddings
    emb0 = get_embedding("Welcome to cat haven")
    emb1 = get_embedding("Discover the joy and companionship of our adorable cats. Join us to find your new feline friend today.")
    emb2 = get_embedding("A small tabby kitten with dark and light brown fur, sitting on a speckled floor, looking downward with a curious expression.")
    emb3 = get_embedding("Tiny yellow dog")
    
    comp0 = cosine_similarity(emb0, emb1)

    comp1 = cosine_similarity(emb1, emb2)
    comp2 = cosine_similarity(emb1, emb3)

    print("comp0", comp0)
    print("comp1", comp1)
    print("comp2", comp2)
main()
"""




@app.local_entrypoint()
def main():
    #html_path = 'file:///Users/nathanlu/code/durable/lab/ranking/_test_good.html'
    #html_path = "https://www.hostgenius.ca"
    html_path = "https://testdurable2.durablesites.com"

    rectangles = scrape(html_path)
    for rect in rectangles:
        print("rect", rect)
        if rect["type"] == "img":
            alt = label(rect["src"])
            emb = get_one_embedding.remote(
                text=alt#rect["alt"]
            )
            #emb = get_one_embedding.remote(image_url=rect["src"])
        else:
            #if rect["type"] == "h1" or rect["type"] == "p" or rect["type"] == "button":
            emb = get_one_embedding.remote(text=rect["content"])
            
        if emb is None:
            continue

        if not isinstance(emb, list):
            emb = emb.tolist()
        rect["embedding"] = emb

    # Save to json
    import json
    with open('sample.json', 'w') as f:
        json.dump(rectangles, f)

