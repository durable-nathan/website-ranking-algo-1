import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.spatial.distance import cosine
from skimage.metrics import structural_similarity as ssim

# Helpers
def calculate_center(bbox):
    x = bbox["x"]
    y = bbox["y"]
    w = bbox["width"]
    h = bbox["height"]

    cx = x + w / 2
    cy = y + h / 2
    return cx, cy

def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p2) - np.array(p1))

def load_data():
    with open('sample.json') as f:
        data = json.load(f)

    return data

def plot_graph(graph):
    fig, ax = plt.subplots()
    for node in graph["nodes"]:
        x = node["x"]
        y = node["y"]
        w = node["width"]
        h = node["height"]

        rect = plt.Rectangle((x, y), w, h, edgecolor='blue', facecolor='none', lw=2)
        ax.add_patch(rect)

    centers = np.array([calculate_center(bbox) for bbox in graph["nodes"]])

    # Plot the centers
    ax.plot(centers[:, 0], centers[:, 1], 'ro')

    for edge in graph["edges"]:
        from_box = edge["from"]
        to_box = edge["to"]
        distance = edge["distance"]
        cosine_similarity = edge["cosine_similarity"]
        score = edge["score"]
        p1 = centers[from_box]
        p2 = centers[to_box]

        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-')
        mid_point = (p1 + p2) / 2
        #ax.text(mid_point[0], mid_point[1], f'{distance:.2f} - {cosine_similarity:.2f}', color='black', fontsize=12, ha='center')
        ax.text(mid_point[0], mid_point[1], f'{cosine_similarity:.2f}', color='black', fontsize=12, ha='center')

    # Set limits and show plot
    ax.set_xlim(0, 1500)
    ax.set_ylim(0, 1500)
    ax.set_aspect('equal')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().invert_yaxis()  # Invert y axis to match the usual coordinate system
    plt.title('Bounding Boxes Distances and Content Similarity Scores')
    plt.grid(True)
    plt.show()

    

def edge_get_content_similarity_score(embeddings1, embeddings2):
    # convert array to numpy
    embeddings1 = np.array(embeddings1[0])
    embeddings2 = np.array(embeddings2[0])

    # Calculate a score based on the cosine similarity
    # of the to and from boxes
    cosine_sim = 1 - cosine(embeddings1, embeddings2)

    return cosine_sim


# Main

def bounding_boxes_to_graph(bboxes):
    graph = {
        "nodes": [bbox for bbox in bboxes],
        "edges": [],
    }
    centers = np.array([calculate_center(bbox) for bbox in bboxes])
    assert len(centers) == len(bboxes)

    # Compute pairwise distances between centers
    num_bboxes = len(centers)
    max_distance = 0
    distances = []

    for i in range(num_bboxes):
        for j in range(i + 1, num_bboxes):
            p1 = centers[i]
            p2 = centers[j]

            distance = euclidean_distance(p1, p2)
            distance_score = 1 - (distance / 1400) 
            distances.append(distance_score)


            # Update max distance if current distance is greater
            if distance_score > max_distance:
                max_distance = distance_score

    # Compute scores based on normalized distances
    for i in range(num_bboxes):
        for j in range(i + 1, num_bboxes):
            p1 = centers[i]
            p2 = centers[j]

            distance = euclidean_distance(p1, p2)
            distance_score = 1 - (distance / 1400) 
            distance_score = distance_score / max_distance if max_distance > 0 else 0

            cosine_similarity = edge_get_content_similarity_score(bboxes[i]["embedding"], bboxes[j]["embedding"])

            score = distance_score * cosine_similarity

            graph["edges"].append({
                "from": i,
                "to": j,
                "distance": distance,
                "distance_score": distance_score,
                "cosine_similarity": cosine_similarity,
                "score": score,
            })

            # Print distances and scores
            print(f"Distance between centers of bounding box {i} and bounding box {j}: {distance:.2f}")
            print(f"Distance score: {distance_score:.2f}")
            print(f"Similarity score: {cosine_similarity:.2f}")
            print(f"Score: {score:.2f}")

    return graph



def calculate_content_similarity_score(graph):
    # Average all the scores

    scores = [edge["cosine_similarity"] for edge in graph["edges"]]
    return sum(scores) / len(scores)

def calculate_symmetry_score(bboxes):
    width = 1440
    height = 1024 

    # Create a matrix of zeros
    matrix = np.zeros((height, width))

    # Fill the matrix with the bounding boxes
    for bbox in bboxes:
        x = bbox["x"]
        y = bbox["y"]
        w = bbox["width"]
        h = bbox["height"]

        matrix[y:y+h, x:x+w] = 1

    # Split into two halves vertically 
    left_half = matrix[:, :width // 2]

    # flip the right half along the y axis so that it mirrors the left half
    right_half = matrix[:, width // 2:]
    right_half = np.fliplr(right_half)

    ssim_index, _ = ssim(left_half, right_half, data_range=1.0, full=True)
    return ssim_index


def main():
    bboxes = load_data()

    # FOR CONTENT COHERENCE
    # Given a list of objects that describe 
    # bounding boxes and their content, generate a graph
    # that represents the similarity between the content
    # (accounts for distance)
    graph = bounding_boxes_to_graph(bboxes)
    print("---")
    cs_score = calculate_content_similarity_score(graph)
    print (f"Content similarity score: {cs_score}")


    # FOR VISUAL BALANCE
    # Given a list of bounding boxes, calculate its symmetry
    sym_score = calculate_symmetry_score(bboxes)
    print(f"Symmetry score: {sym_score}")

    score = (cs_score + sym_score) / 2
    print(f"Content and layout score: {score}")


    # Show
    plot_graph(graph)



main()

