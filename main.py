import tensorflow as tf
from keras.applications import MobileNetV2
from keras.preprocessing  import image
from keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np
import heapq

threat_category='cat'
model = MobileNetV2(weights='imagenet', include_top=True)

def classify_images(images):
    predictions = []
    for img in images:
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        img = tf.image.resize(img, (224, 224))
        
        preds = model.predict(img)
        decoded_preds = decode_predictions(preds, top=3)[0]
        predictions.append(decoded_preds)
    
    return predictions


def mark_cells_with_threat(predictions, threshold=0.2):
    grid = []
    for i in range(0, len(predictions), 4):
        row = []
        for preds in predictions[i:i+4]:
            for _, label, confidence in preds:
                if threat_category in label.lower() and confidence > threshold:
                    row.append(1)  # Mark cell as containing threat 
                    break
            else:
                row.append(0)
        grid.append(row)
    
    assert len(grid) == 4 and all(len(row) == 4 for row in grid), "Grid should have dimensions 4x4"
    return grid

class Node:
    def __init__(self, position, g_value, h_value, parent):
        self.position = position
        self.g_value = g_value
        self.h_value = h_value
        self.parent = parent

    def __lt__(self, other):
        return (self.g_value + self.h_value) < (other.g_value + other.h_value)
    
def heuristic(position, goal):
    return ((position[0] - goal[0]) ** 2 + (position[1] - goal[1]) ** 2) ** 0.5

def get_neighbors(grid, position):
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            neighbor_x = position[0] + dx
            neighbor_y = position[1] + dy
            if 0 <= neighbor_x < len(grid) and 0 <= neighbor_y < len(grid[0]) and grid[neighbor_x][neighbor_y] != 1:
                neighbors.append((neighbor_x, neighbor_y))
    return neighbors

def reconstruct_path(node):
    path = []
    current = node
    while current:
        path.append(current.position)
        current = current.parent
    return list(reversed(path))
    
def astar(grid, start, goal):
    open_list = []
    closed_set = set()
    start_node = Node(start, 0, heuristic(start, goal), None)
    heapq.heappush(open_list, start_node)
    
    while open_list:
        current_node = heapq.heappop(open_list)
        if current_node.position == goal:
            return reconstruct_path(current_node)
        
        closed_set.add(current_node.position)
        
        for neighbor in get_neighbors(grid, current_node.position):
            if neighbor in closed_set:
                continue
            
            g_value = current_node.g_value + 1  # Assuming each step has a cost of 1
            h_value = heuristic(neighbor, goal)
            new_node = Node(neighbor, g_value, h_value, current_node)
            heapq.heappush(open_list, new_node)
    
    return None # No path found

# Usage:
image_array = [
    image.load_img(f'./images/img{i}.jpg') for i in range(1, 17)
]

grid_predictions = classify_images(image_array)
grid = mark_cells_with_threat(grid_predictions)
start = (0, 0)
goal = (3 , 3)
path = astar(grid, start, goal)

print(path)