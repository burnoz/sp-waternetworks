import matplotlib.pyplot as plt

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def show(self):
        return [self.x, self.y]
    

class Node:
    def __init__(self, node_id, x, y, source):
        self.node_id = node_id
        self.position = Point(x, y)
        self.source = source

    def show(self):
        return {
            'node_id': self.node_id,
            'position': self.position.show(),
            'source': self.source
        }
    

class Pipe:
    def __init__(self, start_node, end_node, diameter):
        self.start_node = start_node
        self.end_node = end_node
        self.diameter = diameter
        self.length = getDistance(start_node.position, end_node.position)

    def show(self):
        return {
            'start_node': self.start_node.node_id,
            'end_node': self.end_node.node_id,
            'diameter': self.diameter
        }
    

class NewNode:
    def __init__(self, x, y, diameter):
        self.position = Point(x, y)
        self.diameter = diameter

    def show(self):
        return {
            'position': self.position.show()
        }
    

def getData(file_path):
    nodes = []
    pipes = []
    offices = []
    new_nodes = []
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
    section = None
    for line in lines:
        line = line.strip()
        if line == '[NODES]':
            section = 'nodes'
            continue
        elif line == '[EDGES]':
            section = 'edges'
            continue
        elif line == '[OFFICE]':
            section = 'office'
            continue
        elif line == '[NEW]':
            section = 'new'
            continue
        
        if section == 'nodes':
            parts = line.split()
            node_id = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])
            source = int(parts[3])
            nodes.append(Node(node_id, x, y, source))
        
        elif section == 'edges':
            parts = line.split()
            start_node_id = int(parts[0])
            end_node_id = int(parts[1])
            diameter = float(parts[2])
            
            start_node = next(node for node in nodes if node.node_id == start_node_id)
            end_node = next(node for node in nodes if node.node_id == end_node_id)
            pipes.append(Pipe(start_node, end_node, diameter))
        
        elif section == 'office':
            office_id = int(line)
            offices.append(office_id)
        
        elif section == 'new':
            parts = line.split()
            x = float(parts[0])
            y = float(parts[1])
            diameter = float(parts[2])
            new_nodes.append(NewNode(x, y, diameter))
    
    return nodes, pipes, offices, new_nodes


def getDistance(pointA, pointB):
    return ((pointA.x - pointB.x) ** 2 + (pointA.y - pointB.y) ** 2) ** 0.5

def main():
    nodes, pipes, offices, new_nodes = getData('PES.txt')
    
    # Visualización simple
    plt.figure()
    
    for pipe in pipes:
        x_values = [pipe.start_node.position.x, pipe.end_node.position.x]
        y_values = [pipe.start_node.position.y, pipe.end_node.position.y]
        x_middle = (pipe.start_node.position.x + pipe.end_node.position.x) / 2
        y_middle = (pipe.start_node.position.y + pipe.end_node.position.y) / 2
        plt.plot(x_values, y_values, 'b-')
        plt.text(x_middle, y_middle, str(int(pipe.diameter)), color='blue', fontsize=8)
        plt.text(x_middle, y_middle - 200, f"{pipe.length:.1f}", color='green', fontsize=6)
    
    for node in nodes:
        if node.node_id in offices:
            plt.plot(node.position.x, node.position.y, 'ks')

        else:
            plt.plot(node.position.x, node.position.y, 'ro' if node.source == 1 else 'go')
    
    print("new nodes:", len(new_nodes))
    for new_node in new_nodes:
        plt.plot(new_node.position.x, new_node.position.y, 'kx')
        print("new node at:", new_node.position.show())
    
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Water Network Visualization')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()