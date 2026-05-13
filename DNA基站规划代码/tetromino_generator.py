import random
import json

def generate_tetromino_shape():
    shapes = [
        [(0, 0), (1, 0), (2, 0), (3, 0)],  # 直线
        [(0, 0), (1, 0), (0, 1), (1, 1)],  # 方块
        [(0, 0), (1, 0), (2, 0), (2, 1)],  # L形
        [(0, 0), (1, 0), (1, 1), (2, 1)],  # 反L形
        [(0, 1), (1, 1), (1, 0), (2, 0)],  # S形
        [(0, 0), (1, 0), (1, 1), (2, 1)],  # Z形
        [(0, 0), (0, 1), (1, 1), (1, 0)]   # T形
    ]
    return random.choice(shapes)

def place_tetromino_at(shape, start_x, start_y, occupied_cells, size):
    new_cells = []
    for dx, dy in shape:
        nx, ny = start_x + dx, start_y + dy
        if 0 <= nx < size and 0 <= ny < size and (nx, ny) not in occupied_cells:
            new_cells.append((nx, ny))
        else:
            return False, occupied_cells
    occupied_cells.update(new_cells)
    return True, occupied_cells

def generate_special_area(size, area_ratio=0.15):
    total_cells = size * size
    special_cells_count = int(total_cells * area_ratio)
    occupied_cells = set()

    max_retries = 50
    placed_cells = 0
    retries = 0
    while placed_cells < special_cells_count and retries < max_retries:
        shape = generate_tetromino_shape()
        x = random.randint(0, size - 4)
        y = random.randint(0, size - 4)
        success, occupied_cells = place_tetromino_at(shape, x, y, occupied_cells, size)
        if success:
            placed_cells += len(shape)
            retries = 0
        else:
            retries += 1
        if retries >= max_retries:
            print(f"Max retries reached. Placed {placed_cells} out of {special_cells_count} cells.")
            break
    print(f"Total of {placed_cells} special cells placed.")
    
    save_path = r"D:\桌面\M_PSO\special_area.txt"
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump([[x, y] for x, y in occupied_cells], f, ensure_ascii=False)
    print(f"Special area saved to {save_path}")
    return list(occupied_cells)

from model import canshu
p = canshu()
special_area = generate_special_area(size=p['size'])
