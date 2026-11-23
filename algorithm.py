import heapq
import time
from graphics import update_visualization, reconstruct_path  # Already implemented in graphics.py

def heuristic(cell, goal):
    """
    Computes the heuristic distance between two points using Manhattan distance.
    """
    return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])


def A_star(graph, start, goal, pathMap, ax2, ax3):
    """
    A* algorithm implementation for pathfinding.

    Arguments:
    - graph: The adjacency list representing the maze.
    - start: The starting position (x, y)
    - goal: The goal position (x, y)
    - pathMap: The matrix representation of the maze.
    - ax2: The visualization axis for search progress.
    - ax3: The visualization axis for the final path.
    """
    start_time = time.time()  

    open_list = []
    closed_list = []
    came_from = {}

    g = {start: 0}
    f = {start: heuristic(start, goal)}

    open_list.append((f[start], start))

    expanded = 0

    while open_list:
        open_list.sort()
        current_f, current = open_list.pop(0)

       
        pathMap[current[0], current[1]] = 13
        update_visualization(pathMap, ax2, "Searching...")

        if current == goal:
            final_path = reconstruct_path(came_from, current, pathMap, ax3, "Final Path", start, goal)
            end_time = time.time() - start_time  
            return final_path, expanded, end_time
        
        closed_list.append(current)

        for neighbor in graph[current]:
            if neighbor in closed_list:
                continue

            tentative_g = g.get(current, float('inf')) + 1

            if neighbor not in g or tentative_g < g[neighbor]:
                came_from[neighbor] = current
                g[neighbor] = tentative_g
                f[neighbor] = tentative_g + heuristic(neighbor, goal)

                found_in_open_list = False
                for item in open_list:
                    if item[1] == neighbor:
                        found_in_open_list = True
                        break

                if not found_in_open_list:
                    open_list.append((f[neighbor], neighbor))

                pathMap[neighbor[0], neighbor[1]] = 13
                update_visualization(pathMap, ax2, "Searching...")

        expanded += 1

    end_time = time.time() - start_time  
    return [], expanded, end_time


def bfs(graph, start, goal, pathMap, ax2, ax3):
    """
    Breadth-First Search (BFS) algorithm for pathfinding.

    Arguments:
    - graph: The adjacency list representing the maze.
    - start: The starting position (x, y)
    - goal: The goal position (x, y)
    - pathMap: The matrix representation of the maze.
    - ax2: The visualization axis for search progress.
    - ax3: The visualization axis for the final path.
    """
    start_time = time.time()  

    queue = [start]
    came_from = {}
    visited = [start]

    expanded = 0

    while queue:
        current = queue.pop(0)

        pathMap[current[0], current[1]] = 13
        update_visualization(pathMap, ax2, "Searching...")

        if current == goal:
            final_path = reconstruct_path(came_from, current, pathMap, ax3, "Final Path", start, goal)
            end_time = time.time() - start_time  
            return final_path, expanded, end_time

        for neighbor in graph[current]:
            if neighbor not in visited:
                visited.append(neighbor)
                queue.append(neighbor)
                came_from[neighbor] = current

                pathMap[neighbor[0], neighbor[1]] = 13
                update_visualization(pathMap, ax2, "Searching...")

        expanded += 1

    end_time = time.time() - start_time  
    return [], expanded, end_time


def dfs(graph, start, goal, pathMap, ax2, ax3):
    """
    Depth-First Search (DFS) algorithm for pathfinding.

    Arguments:
    - graph: The adjacency list representing the maze.
    - start: The starting position (x, y)
    - goal: The goal position (x, y)
    - pathMap: The matrix representation of the maze.
    - ax2: The visualization axis for search progress.
    - ax3: The visualization axis for the final path.
    """
    start_time = time.time()  

    stack = [start]
    came_from = {}
    visited = []

    expanded = 0

    while stack:
        current = stack.pop()

        pathMap[current[0], current[1]] = 13
        update_visualization(pathMap, ax2, "Searching...")

        if current == goal:
            final_path = reconstruct_path(came_from, current, pathMap, ax3, "Final Path", start, goal)
            end_time = time.time() - start_time  
            return final_path, expanded, end_time

        visited.append(current)

        for neighbor in graph[current]:
            if neighbor not in visited:
                visited.append(neighbor)
                stack.append(neighbor)
                came_from[neighbor] = current

                pathMap[neighbor[0], neighbor[1]] = 13
                update_visualization(pathMap, ax2, "Searching...")

        expanded += 1

    end_time = time.time() - start_time  
    return [], expanded, end_time


def greedy(graph, start, goal, pathMap, ax2, ax3):
    """
    Greedy Best-First Search algorithm for pathfinding.

    Arguments:
    - graph: The adjacency list representing the maze.
    - start: The starting position (x, y)
    - goal: The goal position (x, y)
    - pathMap: The matrix representation of the maze.
    - ax2: The visualization axis for search progress.
    - ax3: The visualization axis for the final path.
    """
    start_time = time.time()  

    open_list = [start]
    closed_list = []
    came_from = {}

    expanded = 0

    while open_list:
        
        current = min(open_list, key=lambda x: heuristic(x, goal))

        pathMap[current[0], current[1]] = 13
        update_visualization(pathMap, ax2, "Searching...")

        if current == goal:
            final_path = reconstruct_path(came_from, current, pathMap, ax3, "Final Path", start, goal)
            end_time = time.time() - start_time  
            return final_path, expanded, end_time

        open_list.remove(current)
        closed_list.append(current)

        for neighbor in graph[current]:
            if neighbor not in closed_list:
                open_list.append(neighbor)
                came_from[neighbor] = current
                pathMap[neighbor[0], neighbor[1]] = 13
                update_visualization(pathMap, ax2, "Searching...")

        expanded += 1

    end_time = time.time() - start_time  
    return [], expanded, end_time


def depth_limited_search(graph, current, goal, depth, came_from, pathMap, ax2, expanded):
    """
    Depth-Limited Search for pathfinding using a stack.

    Arguments:
    - graph: The adjacency list representing the maze.
    - current: The current position (x, y)
    - goal: The goal position (x, y)
    - depth: The current depth limit
    - came_from: Dictionary tracking visited nodes.
    - pathMap: The matrix representation of the maze.
    - ax2: The visualization axis for search progress.
    - expanded: The count of expanded nodes.
    """
    stack = [(current, 0)] 
    visited = set() 
    while stack:
        current_node, current_depth = stack.pop()  
        pathMap[current_node[0], current_node[1]] = 13
        update_visualization(pathMap, ax2, "Searching...")

        if current_node == goal:
            path = []
            while current_node in came_from:
                path.append(current_node)
                current_node = came_from[current_node]
            path.append((current[0], current[1]))  
            path.reverse()  
            return path, expanded

        if current_depth < depth:
            visited.add(current_node)  
            expanded += 1  
            for neighbor in graph.get(current_node, []):
                if neighbor not in visited:
                    stack.append((neighbor, current_depth + 1))  
                    came_from[neighbor] = current_node  

    return [], expanded  




def iterative_deepening_search(graph, start, goal, pathMap, ax2, ax3):
    """
    Iterative Deepening Search for pathfinding using a stack.

    Arguments:
    - graph: The adjacency list representing the maze.
    - start: The starting position (x, y)
    - goal: The goal position (x, y)
    - pathMap: The matrix representation of the maze.
    - ax2: The visualization axis for search progress.
    - ax3: The visualization axis for the final path.
    """
    start_time = time.time()  
    depth = 0
    total = 0
    while True:
        came_from = {}
       
        result, expanded = depth_limited_search(graph, start, goal, depth, came_from, pathMap, ax2, total)

        total = expanded 

        if result:
            final_path = reconstruct_path(came_from, goal, pathMap, ax3, "Final Path", start, goal)
            end_time = time.time() - start_time  
            return final_path, total, end_time
        depth += 1
