import matplotlib.pyplot as plt
from queue import PriorityQueue
import os

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    

class Node:
    def __init__(self, node_id, x, y, source):
        self.node_id = node_id
        self.position = Point(x, y)
        self.source = source
    

class Pipe:
    def __init__(self, start_node, end_node, diameter):
        self.start_node = start_node
        self.end_node = end_node
        self.diameter = diameter
        self.length = getDistance(start_node.position, end_node.position)
    

class Source:
    def __init__(self, node):
        self.node = node


class NewNode:
    def __init__(self, x, y, diameter):
        self.position = Point(x, y)
        self.diameter = diameter


# Lectura de las instancias
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
    
    # Nodos fuente
    sources = [Source(node) for node in nodes if node.source != 0]
    
    return nodes, pipes, offices, new_nodes, sources

# Calculo de distancia entre dos puntos
def getDistance(pointA, pointB):
    return ((pointA.x - pointB.x) ** 2 + (pointA.y - pointB.y) ** 2) ** 0.5

# Visualizacion de longitudes de tuberias
def pipeLengths(nodes, pipes, offices, new_nodes, show_new_nodes=True):
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Dibujar tuberías con información
    for pipe in pipes:
        x_values = [pipe.start_node.position.x, pipe.end_node.position.x]
        y_values = [pipe.start_node.position.y, pipe.end_node.position.y]
        x_middle = (pipe.start_node.position.x + pipe.end_node.position.x) / 2
        y_middle = (pipe.start_node.position.y + pipe.end_node.position.y) / 2
        ax.plot(x_values, y_values, 'lightgray', linewidth=1.5, alpha=0.6)
        ax.text(x_middle, y_middle, f"D={int(pipe.diameter)}", color='blue', fontsize=7)
        ax.text(x_middle, y_middle - 200, f"L={pipe.length:.1f}", color='green', fontsize=6)
    
    # Dibujar nodos
    for node in nodes:
        x, y = node.position.x, node.position.y
        if node.source == 1:
            ax.plot(x, y, 'o', color='blue', markersize=12,
                    markeredgecolor='black', markeredgewidth=2)
            ax.text(x, y, f'{node.node_id}',
                    fontsize=8, ha='center', va='center', fontweight='bold', color='white')
        elif node.node_id in offices:
            ax.plot(x, y, 's', color='orange', markersize=10,
                    markeredgecolor='black', markeredgewidth=1.5)
            ax.text(x + 150, y, str(node.node_id), fontsize=7, ha='left', va='center')
        else:
            ax.plot(x, y, 'o', color='green', markersize=6,
                    markeredgecolor='black', markeredgewidth=1)
            ax.text(x + 100, y, str(node.node_id), fontsize=6, ha='left', va='center')
    
    # Dibujar nuevos nodos solo si se solicita
    if show_new_nodes and new_nodes:
        for new_node in new_nodes:
            ax.plot(new_node.position.x, new_node.position.y, 'kx',
                    markersize=12, markeredgewidth=2)
    
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('Longitud de Tuberías en la Red de Agua')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# Construccion del grafo con una lista de adyacencia
def buildGraph(nodes, pipes):
    graph = {}

    for node in nodes:
        graph[node.node_id] = []
    
    for pipe in pipes:
        start_id = pipe.start_node.node_id
        end_id = pipe.end_node.node_id
        length = pipe.length
        
        graph[start_id].append((end_id, length, pipe))
        graph[end_id].append((start_id, length, pipe))
    
    return graph

# Dijkstra para encontrar las distancias minimas desde las fuentes
def dijkstra(graph, start_node_id, all_nodes):
    distances = {node.node_id: float('inf') for node in all_nodes}
    distances[start_node_id] = 0
    parent = {node.node_id: None for node in all_nodes}
    
    pq = PriorityQueue()
    pq.put((0, start_node_id))
    
    while not pq.empty():
        current_dist, current_node = pq.get()
        
        if current_dist > distances[current_node]:
            continue
        
        for neighbor, length, pipe in graph[current_node]:
            distance = current_dist + length
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                parent[neighbor] = current_node
                pq.put((distance, neighbor))
    
    return distances, parent

# Sectorizacion de la red
def sectorize(nodes, pipes, sources):
    graph = buildGraph(nodes, pipes)
    node_to_sector = {}
    
    # Calcular Dijkstra desde cada fuente
    source_distances = {}
    for source in sources:
        distances, _ = dijkstra(graph, source.node.node_id, nodes)
        source_distances[source.node.node_id] = distances
    
    # Asignar cada nodo a su fuente más cercana
    for node in nodes:
        min_distance = float('inf')
        closest_source = None
        
        for source in sources:
            distance = source_distances[source.node.node_id][node.node_id]
            
            if distance < min_distance:
                min_distance = distance
                closest_source = source
        
        node_to_sector[node.node_id] = closest_source.node.node_id
    
    # Identificar tuberias que conectan diferentes sectores
    closed_pipes = []
    open_pipes = []
    
    for pipe in pipes:
        start_sector = node_to_sector[pipe.start_node.node_id]
        end_sector = node_to_sector[pipe.end_node.node_id]
        
        if start_sector != end_sector:
            closed_pipes.append(pipe)

        else:
            open_pipes.append(pipe)
    
    return node_to_sector, closed_pipes, open_pipes, source_distances

# Visualizacion de la sectorizacion
def visualizeSectorization(nodes, offices, new_nodes, node_to_sector, closed_pipes, open_pipes, sources, show_new_nodes=True):
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Colores para cada sector
    colors = ['green', 'blue', 'red', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
    sector_colors = {}
    for i, source in enumerate(sources):
        sector_colors[source.node.node_id] = colors[i % len(colors)]
    
    # Dibujar tuberías abiertas
    for pipe in open_pipes:
        x_values = [pipe.start_node.position.x, pipe.end_node.position.x]
        y_values = [pipe.start_node.position.y, pipe.end_node.position.y]
        sector = node_to_sector[pipe.start_node.node_id]
        ax.plot(x_values, y_values, color=sector_colors[sector], linewidth=1, alpha=0.6)
    
    # Dibujar tuberías cerradas
    for pipe in closed_pipes:
        x_values = [pipe.start_node.position.x, pipe.end_node.position.x]
        y_values = [pipe.start_node.position.y, pipe.end_node.position.y]
        ax.plot(x_values, y_values, 'k--', linewidth=3, alpha=0.8, label='Cerrada' if pipe == closed_pipes[0] else '')
    
    # Dibujar nodos
    for node in nodes:
        sector = node_to_sector[node.node_id]
        color = sector_colors[sector]
        x, y = node.position.x, node.position.y

        if node.source == 1:
            ax.plot(x, y, 'o', color=color, markersize=15,
                     markeredgecolor='black', markeredgewidth=2)
            ax.text(x, y, f'{node.node_id}',
                     fontsize=10, ha='center', va='center', fontweight='bold', color='white')
        elif node.node_id in offices:
            ax.plot(x, y, 's', color=color, markersize=10,
                     markeredgecolor='black', markeredgewidth=1.5)
            ax.text(x + 150, y, str(node.node_id), fontsize=7, ha='left', va='center')
        else:
            ax.plot(x, y, 'o', color=color, markersize=8)
            ax.text(x + 100, y, str(node.node_id), fontsize=6, ha='left', va='center')
    
    # Dibujar nuevos nodos solo si se solicita
    if show_new_nodes and new_nodes:
        for new_node in new_nodes:
            ax.plot(new_node.position.x, new_node.position.y, 'kx', markersize=12, markeredgewidth=2)
    
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('Sectorización de la Red de Agua')
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()

# Frescura del agua
def freshnessAnalysis(nodes, node_to_sector, sources, source_distances):
    sector_freshness = {}
    
    for source in sources:
        sector_id = source.node.node_id
        distances = source_distances[sector_id]
        
        max_distance = -1
        farthest_node = None
        
        for node in nodes:
            if node_to_sector[node.node_id] == sector_id:
                distance = distances[node.node_id]

                if distance > max_distance:
                    max_distance = distance
                    farthest_node = node
        
        sector_freshness[sector_id] = {
            'farthest_node': farthest_node.node_id,
            'distance': max_distance
        }
    
    return sector_freshness


def freshnessReport(sector_freshness):
    print("Frescura:")

    for sector_id, data in sector_freshness.items():
        print(f"Sector de la fuente: {sector_id}, nodo más lejano: {data['farthest_node']}, distancia: {data['distance']:.2f}")

    print()

# Flujo máximo de cada sector.
def bfs(graph, source, sink, parent):
    visited = set([source])
    queue = [source]
    
    while queue:
        u = queue.pop(0)
        
        for v in graph[u]:
            # Si no ha sido visitado y hay capacidad residual
            if v not in visited and graph[u].get(v, 0) > 0:
                visited.add(v)
                queue.append(v)

                parent[v] = u

                if v == sink:
                    return True
    
    return False


def edmondsKarp(nodes, pipes, source_id, sink_id):
    # Grafo residual
    graph = {}
    for node in nodes:
        graph[node.node_id] = {}
    
    # Inicializar capacidades
    for pipe in pipes:
        start = pipe.start_node.node_id
        end = pipe.end_node.node_id
        capacity = pipe.diameter
        
        graph[start][end] = capacity
        graph[end][start] = capacity
    
    parent = {}
    max_flow = 0
    flow_graph = {}
    
    # Inicializar flujo de cada tubería en 0
    for node in nodes:
        flow_graph[node.node_id] = {}

        for other_node in nodes:
            flow_graph[node.node_id][other_node.node_id] = 0
    
    # Mientras exista un camino aumentante
    while bfs(graph, source_id, sink_id, parent):
        # Encontrar la capacidad mínima en el camino
        path_flow = float('inf')
        s = sink_id
        
        while s != source_id:
            path_flow = min(path_flow, graph[parent[s]][s])
            s = parent[s]
        
        # Actualizar capacidades residuales y flujo
        max_flow += path_flow
        v = sink_id
        
        while v != source_id:
            u = parent[v]
            graph[u][v] -= path_flow
            graph[v][u] += path_flow
            flow_graph[u][v] += path_flow
            flow_graph[v][u] -= path_flow
            v = parent[v]
        
        parent = {}
    
    return max_flow, flow_graph


def maxFlowAnalysis(nodes, pipes, node_to_sector, sources, sector_freshness):
    results = []
    
    for source in sources:
        sector_id = source.node.node_id
        
        # Obtener nodo más lejano (destino)
        farthest_node_id = sector_freshness[sector_id]['farthest_node']
        
        # Filtrar tuberías del sector
        sector_pipes = [pipe for pipe in pipes 
                       if node_to_sector[pipe.start_node.node_id] == sector_id 
                       and node_to_sector[pipe.end_node.node_id] == sector_id]
        
        # Filtrar nodos del sector
        sector_nodes = [node for node in nodes 
                       if node_to_sector[node.node_id] == sector_id]
        
        # Calcular flujo máximo
        max_flow, flow_graph = edmondsKarp(sector_nodes, sector_pipes, 
                                          sector_id, farthest_node_id)
        
        # Calcular utilización de cada tubería
        pipe_utilization = []
        for pipe in sector_pipes:
            start = pipe.start_node.node_id
            end = pipe.end_node.node_id
            
            # El flujo en la tubería es el máximo de ambas direcciones
            flow = max(abs(flow_graph[start][end]), abs(flow_graph[end][start]))
            capacity = pipe.diameter
            
            if flow > 0:  # Solo mostrar tuberías con flujo
                pipe_utilization.append({
                    'start': start,
                    'end': end,
                    'flow': flow,
                    'capacity': capacity,
                    'percentage': (flow / capacity * 100) if capacity > 0 else 0
                })
        
        results.append({
            'sector_id': sector_id,
            'source': sector_id,
            'sink': farthest_node_id,
            'max_flow': max_flow,
            'pipe_utilization': pipe_utilization
        })
    
    return results


def maxFlowReport(max_flow_results):
    print("Flujo máximo por sector")
    
    for result in max_flow_results:
        print(f"Sector del nodo {result['sector_id']}:")
        print(f"Nodo Origen (fuente): {result['source']}")
        print(f"Sink (más lejano): {result['sink']}")
        print(f"Flujo Máximo: {result['max_flow']:.2f}")
        
        print(f"\nUtilización de tuberías:")
        if result['pipe_utilization']:
            for util in result['pipe_utilization']:
                print(f"Tubería {util['start']} -> {util['end']}: " +
                      f"{util['flow']:.2f} de {util['capacity']:.2f} " +
                      f"({util['percentage']:.1f}%)")
                
            print()
        
        else:
            print("No hay flujo en este sector")


def visualizeMaxFlow(nodes, pipes, node_to_sector, offices, new_nodes, 
                     max_flow_results, sources, show_new_nodes=True):
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Colores para cada sector
    colors = ['green', 'blue', 'red', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
    sector_colors = {}
    for i, source in enumerate(sources):
        sector_colors[source.node.node_id] = colors[i % len(colors)]
    
    # Dibujar tuberías con grosor proporcional al flujo
    for result in max_flow_results:
        sector_id = result['sector_id']
        color = sector_colors[sector_id]
        
        for util in result['pipe_utilization']:
            # Encontrar la tubería
            pipe = next((p for p in pipes 
                        if (p.start_node.node_id == util['start'] and 
                            p.end_node.node_id == util['end']) or
                           (p.start_node.node_id == util['end'] and 
                            p.end_node.node_id == util['start'])), None)
            
            if pipe:
                x_values = [pipe.start_node.position.x, pipe.end_node.position.x]
                y_values = [pipe.start_node.position.y, pipe.end_node.position.y]
                
                # Grosor proporcional al porcentaje de utilización
                linewidth = 1 + (util['percentage'] / 100) * 5
                
                ax.plot(x_values, y_values, color=color, 
                        linewidth=linewidth, alpha=0.7)
                
                # Mostrar flujo en el medio
                x_mid = (x_values[0] + x_values[1]) / 2
                y_mid = (y_values[0] + y_values[1]) / 2
                ax.text(x_mid, y_mid, f"{util['flow']}/{util['capacity']}", 
                        fontsize=6, ha='center', 
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
    
    # Dibujar tuberías sin flujo en gris claro
    for pipe in pipes:
        has_flow = False
        for result in max_flow_results:
            if any(u['start'] == pipe.start_node.node_id and u['end'] == pipe.end_node.node_id or
                   u['start'] == pipe.end_node.node_id and u['end'] == pipe.start_node.node_id
                   for u in result['pipe_utilization']):
                has_flow = True
                break
        
        if not has_flow:
            x_values = [pipe.start_node.position.x, pipe.end_node.position.x]
            y_values = [pipe.start_node.position.y, pipe.end_node.position.y]
            ax.plot(x_values, y_values, 'lightgray', linewidth=1, alpha=0.3)
    
    # Dibujar nodos
    for node in nodes:
        sector = node_to_sector[node.node_id]
        color = sector_colors[sector]
        x, y = node.position.x, node.position.y
        
        is_sink = any(r['sink'] == node.node_id for r in max_flow_results)
        
        if node.source == 1:  # Fuente
            ax.plot(x, y, 'o', color=color, markersize=15,
                    markeredgecolor='black', markeredgewidth=2)
            ax.text(x, y, f'{node.node_id}',
                    fontsize=10, ha='center', va='center', color='white', fontweight='bold')
        elif is_sink:  # Nodo más lejano
            ax.plot(x, y, '*', color=color, markersize=18,
                    markeredgecolor='black')
            ax.text(x + 150, y, f'{node.node_id}',
                    fontsize=7, ha='left', 
                    bbox=dict(alpha=0.7, pad=2))
        elif node.node_id in offices:
            ax.plot(x, y, 's', color=color, markersize=10,
                    markeredgecolor='black', markeredgewidth=1.5)
        else:
            ax.plot(x, y, 'o', color=color, markersize=6)
            ax.text(x, y, str(node.node_id), fontsize=6, ha='center', va='center', color='white')
    
    # Dibujar nuevos nodos solo si se solicita
    if show_new_nodes and new_nodes:
        for new_node in new_nodes:
            ax.plot(new_node.position.x, new_node.position.y, 'kx', 
                    markersize=12, markeredgewidth=2)
    
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('Flujo Máximo por Sector')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# Muestras de calidad del agua
def tspNearestNeighbor(nodes, pipes, start_node_id):
    graph = buildGraph(nodes, pipes)
    unvisited = set(node.node_id for node in nodes)
    current = start_node_id
    route = [current]
    total_distance = 0
    distances = []
    
    unvisited.remove(current)
    
    while unvisited:
        # Calcular distancias desde el nodo actual usando Dijkstra
        dist_from_current, parent = dijkstra(graph, current, nodes)
        
        # Encontrar el nodo no visitado más cercano
        nearest = None
        min_dist = float('inf')
        
        for node_id in unvisited:
            if dist_from_current[node_id] < min_dist:
                min_dist = dist_from_current[node_id]
                nearest = node_id
        
        route.append(nearest)
        distances.append(min_dist)
        total_distance += min_dist
        current = nearest
        unvisited.remove(nearest)
    
    # Regresar al punto de inicio
    dist_to_start, _ = dijkstra(graph, current, nodes)
    final_dist = dist_to_start[start_node_id]
    route.append(start_node_id)
    distances.append(final_dist)
    total_distance += final_dist
    
    return route, distances, total_distance


def waterSamplingAnalysis(nodes, pipes, offices):
    results = []
    
    for office_id in offices:
        print(f"Ruta de muestreo desde nodo {office_id} (oficina):")
        
        route, distances, total_distance = tspNearestNeighbor(nodes, pipes, office_id)
        
        results.append({
            'office_id': office_id,
            'route': route,
            'distances': distances,
            'total_distance': total_distance
        })
    
    return results


def visualizeSampling(nodes, pipes, offices, new_nodes, sampling_results, show_new_nodes=True):
    graph = buildGraph(nodes, pipes)
    
    for result in sampling_results:
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Dibujar todas las tuberías en gris claro
        for pipe in pipes:
            x_values = [pipe.start_node.position.x, pipe.end_node.position.x]
            y_values = [pipe.start_node.position.y, pipe.end_node.position.y]
            ax.plot(x_values, y_values, 'lightgray', linewidth=1, alpha=0.4)
        
        # Dibujar la ruta de muestreo usando caminos reales
        route = result['route']
        segment_number = 1
        
        for i in range(len(route) - 1):
            start_id = route[i]
            end_id = route[i + 1]
            
            # Usar Dijkstra para encontrar el camino real entre estos dos nodos
            distances, parent = dijkstra(graph, start_id, nodes)
            
            # Reconstruir el camino
            path = []
            current = end_id
            while current is not None:
                path.append(current)
                current = parent[current]
            path.reverse()
            
            # Dibujar cada segmento del camino
            color_intensity = i / (len(route) - 1) if len(route) > 1 else 0
            color = plt.cm.rainbow(color_intensity)
            
            for j in range(len(path) - 1):
                node1 = next(n for n in nodes if n.node_id == path[j])
                node2 = next(n for n in nodes if n.node_id == path[j + 1])
                
                x_values = [node1.position.x, node2.position.x]
                y_values = [node1.position.y, node2.position.y]
                
                ax.plot(x_values, y_values, color=color, linewidth=3, alpha=0.8)
                
                # Flecha en el primer segmento de cada tramo
                if j == 0:
                    dx = x_values[1] - x_values[0]
                    dy = y_values[1] - y_values[0]
                    ax.arrow(x_values[0] + dx*0.4, y_values[0] + dy*0.4,
                             dx*0.2, dy*0.2, head_width=100, head_length=80,
                             fc=color, ec=color, alpha=0.7)
            
            # Mostrar número de segmento en el punto medio del camino
            if len(path) > 1:
                mid_idx = len(path) // 2
                node_mid = next(n for n in nodes if n.node_id == path[mid_idx])
                ax.text(node_mid.position.x, node_mid.position.y, str(segment_number),
                        fontsize=9, ha='center', va='center', fontweight='bold',
                        bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', linewidth=1.5, pad=3))
                segment_number += 1
        
        # Dibujar nodos
        for node in nodes:
            x, y = node.position.x, node.position.y
            
            if node.node_id == result['office_id']:
                # Oficina (punto de inicio/fin)
                ax.plot(x, y, 's', color='red', markersize=15,
                        markeredgecolor='black', markeredgewidth=2)
                ax.text(x, y - 300, f'Oficina\n{node.node_id}',
                        fontsize=9, ha='center', va='top',
                        fontweight='bold',
                        bbox=dict(alpha=0.9, edgecolor='black', pad=3))
            elif node.source == 1:
                # Fuente
                ax.plot(x, y, 'o', color='blue', markersize=12,
                        markeredgecolor='black', markeredgewidth=2)
                ax.text(x, y, f'{node.node_id}',
                        fontsize=8, ha='center', va='center', fontweight='bold', color='white')
            elif node.node_id in offices:
                ax.plot(x, y, 's', color='orange', markersize=10,
                        markeredgecolor='black', markeredgewidth=1.5)
            else:
                # Nodo regular
                ax.plot(x, y, 'o', color='green', markersize=6,
                        markeredgecolor='black', markeredgewidth=1)
        
        # Dibujar nuevos nodos solo si se solicita
        if show_new_nodes and new_nodes:
            for new_node in new_nodes:
                ax.plot(new_node.position.x, new_node.position.y, 'kx',
                        markersize=12, markeredgewidth=2)
        
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title(f'Ruta de Muestreo desde Oficina {result["office_id"]}\n' +
                     f'Distancia Total: {result["total_distance"]:.2f} unidades')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def samplingReport(sampling_results, nodes, pipes):
    graph = buildGraph(nodes, pipes)
    
    for result in sampling_results:
        print(f"Distancia Total: {result['total_distance']:.2f} unidades")
        print()
        
        # Construir el recorrido completo con TODOS los nodos
        route = result['route']
        complete_route = []
              
        for i in range(len(route) - 1):
            start_id = route[i]
            end_id = route[i + 1]
            distance = result['distances'][i]
            
            # Usar Dijkstra para encontrar el camino real
            distances_map, parent = dijkstra(graph, start_id, nodes)
            
            # Reconstruir el camino completo
            path = []
            current = end_id
            while current is not None:
                path.append(current)
                current = parent[current]
            path.reverse()
            
            # Agregar al recorrido completo (evitando duplicados en las uniones)
            if i == 0:
                complete_route.extend(path)
            else:
                complete_route.extend(path[1:])  # Omitir el primer nodo que ya está
        
        print(f"Orden para recoger muestras:")
        
        # Mostrar ruta de muestreo con costos
        route_with_costs = []
        for i in range(len(route)):
            if i == 0:
                route_with_costs.append(str(route[i]))
            else:
                route_with_costs.append(f"-({result['distances'][i-1]:.2f})-> {route[i]}")
        print(' '.join(route_with_costs))
        
        print(f"\nRecorrido completo:")
        # Construir recorrido completo con costos
        complete_with_costs = []
        graph_built = buildGraph(nodes, pipes)
        
        for i in range(len(complete_route)):
            if i == 0:
                complete_with_costs.append(str(complete_route[i]))
            else:
                node1 = next(n for n in nodes if n.node_id == complete_route[i-1])
                node2 = next(n for n in nodes if n.node_id == complete_route[i])
                edge_dist = getDistance(node1.position, node2.position)
                complete_with_costs.append(f"-({edge_dist:.2f})-> {complete_route[i]}")
        
        print(' '.join(complete_with_costs))
    
    print()


# 7.- Conexión de nuevos nodos a la red
def connectNewNodes(nodes, pipes, new_nodes):
    updated_nodes = nodes[:]
    updated_pipes = pipes[:]
    new_connections = []
    
    # Ultimo id usado
    max_node_id = max(node.node_id for node in nodes)
    
    for i, new_node in enumerate(new_nodes):
        new_node_id = max_node_id + i + 1
        new_node_obj = Node(new_node_id, new_node.position.x, new_node.position.y, source=0)
        
        min_distance = float('inf')
        closest_node = None
        
        for node in nodes:
            # No fuentes
            if node.source == 0:
                distance = getDistance(new_node.position, node.position)

                if distance < min_distance:
                    min_distance = distance
                    closest_node = node
        
        if closest_node:
            # Nueva tuberia con el diametro especificado
            new_pipe = Pipe(new_node_obj, closest_node, new_node.diameter)
            updated_pipes.append(new_pipe)
            
            new_connections.append({
                'new_node_id': new_node_id,
                'connected_to': closest_node.node_id,
                'distance': min_distance,
                'diameter': new_node.diameter
            })
        
        updated_nodes.append(new_node_obj)
    
    return updated_nodes, updated_pipes, new_connections


def newNodesReport(new_connections):
    print("Nodos nuevos:")
    
    for conn in new_connections:
        print(f"Nodo {conn['new_node_id']}:")
        print(f"Conectado al nodo: {conn['connected_to']}")
        print(f"Longitud de la tubería: {conn['distance']:.2f} unidades")
        print(f"Diámetro: {conn['diameter']:.2f}")
        print()


def visualizeNewConnections(nodes, pipes, offices, new_connections, title_suffix=""):
    plt.figure(figsize=(14, 12))
    
    # Dibujar todas las tuberías existentes en gris
    for pipe in pipes:
        # Verificar si es una tubería nueva
        is_new = any(
            (conn['new_node_id'] == pipe.start_node.node_id or 
             conn['new_node_id'] == pipe.end_node.node_id)
            for conn in new_connections
        )
        
        x_values = [pipe.start_node.position.x, pipe.end_node.position.x]
        y_values = [pipe.start_node.position.y, pipe.end_node.position.y]
        
        if is_new:
            # Tuberías nuevas en rojo grueso
            plt.plot(x_values, y_values, 'r-', linewidth=3, alpha=0.8, label='Nueva' if pipe == pipes[-1] else '')
            
            # Mostrar longitud y diámetro
            x_mid = (x_values[0] + x_values[1]) / 2
            y_mid = (y_values[0] + y_values[1]) / 2
            plt.text(x_mid, y_mid, f"L={pipe.length:.2f}\nD={pipe.diameter}",
                    fontsize=8, ha='center', va='center',
                    bbox=dict(alpha=0.8, pad=2, edgecolor='black'))
        else:
            plt.plot(x_values, y_values, 'gray', linewidth=1, alpha=0.4)
    
    # Dibujar nodos existentes
    for node in nodes:
        x, y = node.position.x, node.position.y
        
        if node.source == 1:
            plt.plot(x, y, 'o', color='blue', markersize=15,
                    markeredgecolor='black', markeredgewidth=2)
            plt.text(x, y, f'{node.node_id}',
                    fontsize=10, ha='center', va='center', color='white')
        elif node.node_id in offices:
            plt.plot(x, y, 's', color='orange', markersize=12,
                    markeredgecolor='black', markeredgewidth=2)
            plt.text(x + 200, y, str(node.node_id), fontsize=8, ha='left', va='center')
        else:
            plt.plot(x, y, 'o', color='green', markersize=8,
                    markeredgecolor='black', markeredgewidth=1)
            plt.text(x + 150, y, str(node.node_id), fontsize=7, ha='left', va='center')
    
    # Dibujar nuevos nodos conectados
    for conn in new_connections:
        new_node = next(n for n in nodes if n.node_id == conn['new_node_id'])
        x, y = new_node.position.x, new_node.position.y
        
        plt.plot(x, y, '*', color='red', markersize=20,
                markeredgecolor='black', markeredgewidth=2)
    
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(f'Red de Agua con nuevos nodos {title_suffix}')
    plt.grid(alpha=0.3)
    if new_connections:
        plt.legend()
    plt.tight_layout()
    plt.show()


def saveReport(filepath, instance_name, content):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"REPORTE DE ANÁLISIS DE RED DE DISTRIBUCIÓN DE AGUA\n")
        f.write(f"{'='*70}\n")
        f.write(f"Instancia: {instance_name}\n")
        f.write(f"{'='*70}\n\n")
        f.write(content)


def generateReport(instance_name, nodes, pipes, offices, node_to_sector, closed_pipes, 
                   sector_freshness, max_flow_results, sampling_results, sources, is_after=False):
    report = []
    
    # Problema 1: Información de la red
    report.append("[PROBLEMA 1: ESTRUCTURA DE LA RED]\n")
    report.append(f"Número de nodos: {len(nodes)}\n")
    report.append(f"Número de tuberías: {len(pipes)}\n")
    report.append(f"Número de fuentes: {len(sources)}\n")
    
    report.append("\nNodos fuente:\n")
    for source in sources:
        report.append(f"  - Nodo {source.node.node_id}\n")
    report.append("\n")
    
    report.append("Oficinas:\n")
    for office in offices:
        report.append(f"  - Nodo {office}\n")
    report.append("\n")
    
    # Problema 2: Longitudes de tuberías
    report.append("[PROBLEMA 2: LONGITUDES Y DIÁMETROS DE TUBERÍAS]\n")
    report.append(f"{'Inicio':<10}{'Fin':<10}{'Longitud':<15}{'Diámetro':<15}\n")
    report.append(f"{'-'*50}\n")

    for pipe in pipes:
        report.append(f"{pipe.start_node.node_id:<10}{pipe.end_node.node_id:<10}"
                     f"{pipe.length:<15.2f}{pipe.diameter:<15.2f}\n")
        
    report.append("\n")
    
    # Problema 3: Sectorización
    report.append("[PROBLEMA 3: SECTORIZACIÓN DE LA RED]\n")
    for source in sources:
        sector_id = source.node.node_id
        sector_nodes = [n.node_id for n in nodes if node_to_sector[n.node_id] == sector_id]
        report.append(f"Sector de la fuente {sector_id}:\n")
        report.append(f"  Nodos en el sector: {sector_nodes}\n")
        report.append(f"  Total de nodos: {len(sector_nodes)}\n\n")
    
    report.append(f"Tuberías cerradas (entre sectores): {len(closed_pipes)}\n")
    if closed_pipes:
        for pipe in closed_pipes:
            report.append(f"  - Tubería {pipe.start_node.node_id} <-> {pipe.end_node.node_id}\n")
    report.append("\n")
    
    # Problema 4: Frescura del agua
    report.append("[PROBLEMA 4: FRESCURA DEL AGUA]\n")
    for sector_id, data in sector_freshness.items():
        report.append(f"Sector de la fuente {sector_id}:\n")
        report.append(f"  Nodo más lejano: {data['farthest_node']}\n")
        report.append(f"  Distancia: {data['distance']:.2f} unidades\n\n")
    
    # Problema 5: Flujo máximo
    report.append("[PROBLEMA 5: FLUJO MÁXIMO POR SECTOR]\n")
    for result in max_flow_results:
        report.append(f"Sector del nodo {result['sector_id']}:\n")
        report.append(f"  Nodo origen (fuente): {result['source']}\n")
        report.append(f"  Nodo destino (más lejano): {result['sink']}\n")
        report.append(f"  Flujo máximo: {result['max_flow']:.2f}\n\n")
        
        if result['pipe_utilization']:
            report.append(f"  Utilización de tuberías:\n")
            for util in result['pipe_utilization']:
                report.append(f"    Tubería {util['start']} -> {util['end']}: "
                            f"{util['flow']:.2f} de {util['capacity']:.2f} "
                            f"({util['percentage']:.1f}%)\n")
        else:
            report.append(f"  No hay flujo en este sector\n")
        report.append("\n")
    
    # Problema 6: Muestras de agua
    report.append("[PROBLEMA 6: RUTAS DE MUESTREO DE CALIDAD DEL AGUA]\n")
    
    # Construir grafo para reconstruir caminos completos
    graph = buildGraph(nodes, pipes)
    
    for result in sampling_results:
        report.append(f"\nOficina {result['office_id']}:\n")
        report.append(f"  Distancia total: {result['total_distance']:.2f} unidades\n\n")
        
        # Orden para recoger muestras (solo nodos principales)
        route = result['route']
        report.append(f"  Orden para recoger muestras:\n  ")
        route_with_costs = []
        for i in range(len(route)):
            if i == 0:
                route_with_costs.append(str(route[i]))
            else:
                route_with_costs.append(f"-({result['distances'][i-1]:.2f})-> {route[i]}")
        report.append(' '.join(route_with_costs))
        report.append("\n\n")
        
        # Recorrido completo (con todos los nodos intermedios)
        complete_route = []
        
        for i in range(len(route) - 1):
            start_id = route[i]
            end_id = route[i + 1]
            
            # Usar Dijkstra para encontrar el camino real
            distances_map, parent = dijkstra(graph, start_id, nodes)
            
            # Reconstruir el camino completo
            path = []
            current = end_id
            while current is not None:
                path.append(current)
                current = parent[current]
            path.reverse()
            
            # Agregar al recorrido completo (evitando duplicados en las uniones)
            if i == 0:
                complete_route.extend(path)
            else:
                complete_route.extend(path[1:])  # Omitir el primer nodo que ya está
        
        # Construir recorrido completo con costos
        report.append(f"  Recorrido completo:\n  ")
        complete_with_costs = []
        
        for i in range(len(complete_route)):
            if i == 0:
                complete_with_costs.append(str(complete_route[i]))
            else:
                node1 = next(n for n in nodes if n.node_id == complete_route[i-1])
                node2 = next(n for n in nodes if n.node_id == complete_route[i])
                edge_dist = getDistance(node1.position, node2.position)
                complete_with_costs.append(f"-({edge_dist:.2f})-> {complete_route[i]}")
        
        report.append(' '.join(complete_with_costs))
        report.append("\n")

    return ''.join(report)


def saveFigure(filepath, fig=None):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if fig is None:
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
    else:
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()


def pipeLengthsWithSave(nodes, pipes, offices, new_nodes, output_path, show_new_nodes=True):
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Dibujar tuberías con información
    for pipe in pipes:
        x_values = [pipe.start_node.position.x, pipe.end_node.position.x]
        y_values = [pipe.start_node.position.y, pipe.end_node.position.y]
        x_middle = (pipe.start_node.position.x + pipe.end_node.position.x) / 2
        y_middle = (pipe.start_node.position.y + pipe.end_node.position.y) / 2
        ax.plot(x_values, y_values, 'lightgray', linewidth=1.5, alpha=0.6)
        ax.text(x_middle, y_middle, f"D={int(pipe.diameter)}", color='blue', fontsize=7)
        ax.text(x_middle, y_middle - 200, f"L={pipe.length:.1f}", color='green', fontsize=6)
    
    # Dibujar nodos
    for node in nodes:
        x, y = node.position.x, node.position.y
        if node.source == 1:
            ax.plot(x, y, 'o', color='blue', markersize=12,
                    markeredgecolor='black', markeredgewidth=2)
            ax.text(x, y, f'{node.node_id}',
                    fontsize=8, ha='center', va='center', fontweight='bold', color='white')
        elif node.node_id in offices:
            ax.plot(x, y, 's', color='orange', markersize=10,
                    markeredgecolor='black', markeredgewidth=1.5)
            ax.text(x + 150, y, str(node.node_id), fontsize=7, ha='left', va='center')
        else:
            ax.plot(x, y, 'o', color='green', markersize=6,
                    markeredgecolor='black', markeredgewidth=1)
            ax.text(x + 100, y, str(node.node_id), fontsize=6, ha='left', va='center')
    
    if show_new_nodes and new_nodes:
        for new_node in new_nodes:
            ax.plot(new_node.position.x, new_node.position.y, 'kx',
                    markersize=12, markeredgewidth=2)
    
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('Longitud de Tuberías en la Red de Agua')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    saveFigure(output_path)


def visualizeSectorizationWithSave(nodes, offices, new_nodes, node_to_sector, closed_pipes, 
                                   open_pipes, sources, output_path, show_new_nodes=True):
    fig, ax = plt.subplots(figsize=(12, 10))
    
    colors = ['green', 'blue', 'red', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
    sector_colors = {}
    for i, source in enumerate(sources):
        sector_colors[source.node.node_id] = colors[i % len(colors)]
    
    for pipe in open_pipes:
        x_values = [pipe.start_node.position.x, pipe.end_node.position.x]
        y_values = [pipe.start_node.position.y, pipe.end_node.position.y]
        sector = node_to_sector[pipe.start_node.node_id]
        ax.plot(x_values, y_values, color=sector_colors[sector], linewidth=1, alpha=0.6)
    
    for pipe in closed_pipes:
        x_values = [pipe.start_node.position.x, pipe.end_node.position.x]
        y_values = [pipe.start_node.position.y, pipe.end_node.position.y]
        ax.plot(x_values, y_values, 'k--', linewidth=3, alpha=0.8, label='Cerrada' if pipe == closed_pipes[0] else '')
    
    # Dibujar nodos
    for node in nodes:
        sector = node_to_sector[node.node_id]
        color = sector_colors[sector]
        x, y = node.position.x, node.position.y

        if node.source == 1:
            ax.plot(x, y, 'o', color=color, markersize=15,
                     markeredgecolor='black', markeredgewidth=2)
            ax.text(x, y, f'{node.node_id}',
                     fontsize=10, ha='center', va='center', fontweight='bold', color='white')
        elif node.node_id in offices:
            ax.plot(x, y, 's', color=color, markersize=10,
                     markeredgecolor='black', markeredgewidth=1.5)
            ax.text(x + 150, y, str(node.node_id), fontsize=7, ha='left', va='center')
        else:
            ax.plot(x, y, 'o', color=color, markersize=8)
            ax.text(x + 100, y, str(node.node_id), fontsize=6, ha='left', va='center')
    
    # Dibujar nuevos nodos solo si se solicita
    if show_new_nodes and new_nodes:
        for new_node in new_nodes:
            ax.plot(new_node.position.x, new_node.position.y, 'kx', markersize=12, markeredgewidth=2)
    
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('Sectorización de la Red de Agua')
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    saveFigure(output_path)


def visualizeMaxFlowWithSave(nodes, pipes, node_to_sector, offices, new_nodes, 
                             max_flow_results, sources, output_path, show_new_nodes=True):
    fig, ax = plt.subplots(figsize=(12, 10))
    
    colors = ['green', 'blue', 'red', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
    sector_colors = {}
    for i, source in enumerate(sources):
        sector_colors[source.node.node_id] = colors[i % len(colors)]
    
    for result in max_flow_results:
        sector_id = result['sector_id']
        color = sector_colors[sector_id]
        
        for util in result['pipe_utilization']:
            pipe = next((p for p in pipes 
                        if (p.start_node.node_id == util['start'] and 
                            p.end_node.node_id == util['end']) or
                           (p.start_node.node_id == util['end'] and 
                            p.end_node.node_id == util['start'])), None)
            
            if pipe:
                x_values = [pipe.start_node.position.x, pipe.end_node.position.x]
                y_values = [pipe.start_node.position.y, pipe.end_node.position.y]
                linewidth = 1 + (util['percentage'] / 100) * 5
                
                ax.plot(x_values, y_values, color=color, 
                        linewidth=linewidth, alpha=0.7)
                
                x_mid = (x_values[0] + x_values[1]) / 2
                y_mid = (y_values[0] + y_values[1]) / 2
                ax.text(x_mid, y_mid, f"{util['flow']}/{util['capacity']}", 
                        fontsize=6, ha='center', 
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
    
    # Dibujar tuberías sin flujo en gris claro
    for pipe in pipes:
        has_flow = False
        for result in max_flow_results:
            if any(u['start'] == pipe.start_node.node_id and u['end'] == pipe.end_node.node_id or
                   u['start'] == pipe.end_node.node_id and u['end'] == pipe.start_node.node_id
                   for u in result['pipe_utilization']):
                has_flow = True
                break
        
        if not has_flow:
            x_values = [pipe.start_node.position.x, pipe.end_node.position.x]
            y_values = [pipe.start_node.position.y, pipe.end_node.position.y]
            ax.plot(x_values, y_values, 'lightgray', linewidth=1, alpha=0.3)
    
    # Dibujar nodos
    for node in nodes:
        sector = node_to_sector[node.node_id]
        color = sector_colors[sector]
        x, y = node.position.x, node.position.y
        
        is_sink = any(r['sink'] == node.node_id for r in max_flow_results)
        
        if node.source == 1:  # Fuente
            ax.plot(x, y, 'o', color=color, markersize=15,
                    markeredgecolor='black', markeredgewidth=2)
            ax.text(x, y, f'{node.node_id}',
                    fontsize=10, ha='center', va='center', color='white', fontweight='bold')
        elif is_sink:  # Nodo más lejano
            ax.plot(x, y, '*', color=color, markersize=18,
                    markeredgecolor='black')
            ax.text(x + 150, y, f'{node.node_id}',
                    fontsize=7, ha='left', 
                    bbox=dict(alpha=0.7, pad=2))
        elif node.node_id in offices:
            ax.plot(x, y, 's', color=color, markersize=10,
                    markeredgecolor='black', markeredgewidth=1.5)
        else:
            ax.plot(x, y, 'o', color=color, markersize=6)
            ax.text(x, y, str(node.node_id), fontsize=6, ha='center', va='center', color='white')
    
    # Dibujar nuevos nodos solo si se solicita
    if show_new_nodes and new_nodes:
        for new_node in new_nodes:
            ax.plot(new_node.position.x, new_node.position.y, 'kx', 
                    markersize=12, markeredgewidth=2)
    
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('Flujo Máximo por Sector')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    saveFigure(output_path)


def visualizeSamplingWithSave(nodes, pipes, offices, new_nodes, sampling_results, 
                              output_path_prefix, show_new_nodes=True):
    graph = buildGraph(nodes, pipes)
    
    for idx, result in enumerate(sampling_results):
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Dibujar todas las tuberías en gris claro
        for pipe in pipes:
            x_values = [pipe.start_node.position.x, pipe.end_node.position.x]
            y_values = [pipe.start_node.position.y, pipe.end_node.position.y]
            ax.plot(x_values, y_values, 'lightgray', linewidth=1, alpha=0.4)
        
        # Dibujar la ruta de muestreo usando caminos reales
        route = result['route']
        segment_number = 1
        
        for i in range(len(route) - 1):
            start_id = route[i]
            end_id = route[i + 1]
            
            # Usar Dijkstra para encontrar el camino real entre estos dos nodos
            distances, parent = dijkstra(graph, start_id, nodes)
            
            # Reconstruir el camino
            path = []
            current = end_id
            while current is not None:
                path.append(current)
                current = parent[current]
            path.reverse()
            
            # Dibujar cada segmento del camino
            color_intensity = i / (len(route) - 1) if len(route) > 1 else 0
            color = plt.cm.rainbow(color_intensity)
            
            for j in range(len(path) - 1):
                node1 = next(n for n in nodes if n.node_id == path[j])
                node2 = next(n for n in nodes if n.node_id == path[j + 1])
                
                x_values = [node1.position.x, node2.position.x]
                y_values = [node1.position.y, node2.position.y]
                
                ax.plot(x_values, y_values, color=color, linewidth=3, alpha=0.8)
                
                # Flecha en el primer segmento de cada tramo
                if j == 0:
                    dx = x_values[1] - x_values[0]
                    dy = y_values[1] - y_values[0]
                    ax.arrow(x_values[0] + dx*0.4, y_values[0] + dy*0.4,
                             dx*0.2, dy*0.2, head_width=100, head_length=80,
                             fc=color, ec=color, alpha=0.7)
            
            # Mostrar número de segmento en el punto medio del camino
            if len(path) > 1:
                mid_idx = len(path) // 2
                node_mid = next(n for n in nodes if n.node_id == path[mid_idx])
                ax.text(node_mid.position.x, node_mid.position.y, str(segment_number),
                        fontsize=9, ha='center', va='center', fontweight='bold',
                        bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', linewidth=1.5, pad=3))
                segment_number += 1
        
        # Dibujar nodos
        for node in nodes:
            x, y = node.position.x, node.position.y
            
            if node.node_id == result['office_id']:
                # Oficina (punto de inicio/fin)
                ax.plot(x, y, 's', color='red', markersize=15,
                        markeredgecolor='black', markeredgewidth=2)
                ax.text(x, y - 300, f'Oficina\n{node.node_id}',
                        fontsize=9, ha='center', va='top',
                        fontweight='bold',
                        bbox=dict(alpha=0.9, edgecolor='black', pad=3))
            elif node.source == 1:
                # Fuente
                ax.plot(x, y, 'o', color='blue', markersize=12,
                        markeredgecolor='black', markeredgewidth=2)
                ax.text(x, y, f'{node.node_id}',
                        fontsize=8, ha='center', va='center', fontweight='bold', color='white')
            elif node.node_id in offices:
                ax.plot(x, y, 's', color='orange', markersize=10,
                        markeredgecolor='black', markeredgewidth=1.5)
            else:
                # Nodo regular
                ax.plot(x, y, 'o', color='green', markersize=6,
                        markeredgecolor='black', markeredgewidth=1)
        
        # Dibujar nuevos nodos solo si se solicita
        if show_new_nodes and new_nodes:
            for new_node in new_nodes:
                ax.plot(new_node.position.x, new_node.position.y, 'kx',
                        markersize=12, markeredgewidth=2)
        
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title(f'Ruta de Muestreo desde Oficina {result["office_id"]}\n' +
                     f'Distancia Total: {result["total_distance"]:.2f} unidades')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        output_file = f"{output_path_prefix}_oficina_{result['office_id']}.png"
        saveFigure(output_file)


def visualizeNewConnectionsWithSave(nodes, pipes, offices, new_connections, output_path, title_suffix=""):
    fig = plt.figure(figsize=(14, 12))
    
    for pipe in pipes:
        is_new = any(
            (conn['new_node_id'] == pipe.start_node.node_id or 
             conn['new_node_id'] == pipe.end_node.node_id)
            for conn in new_connections
        )
        
        x_values = [pipe.start_node.position.x, pipe.end_node.position.x]
        y_values = [pipe.start_node.position.y, pipe.end_node.position.y]
        
        if is_new:
            plt.plot(x_values, y_values, 'r-', linewidth=3, alpha=0.8, label='Nueva' if pipe == pipes[-1] else '')
            
            # Mostrar longitud y diámetro
            x_mid = (x_values[0] + x_values[1]) / 2
            y_mid = (y_values[0] + y_values[1]) / 2
            plt.text(x_mid, y_mid, f"L={pipe.length:.2f}\nD={pipe.diameter}",
                    fontsize=8, ha='center', va='center',
                    bbox=dict(alpha=0.8, pad=2, edgecolor='black'))
        else:
            plt.plot(x_values, y_values, 'gray', linewidth=1, alpha=0.4)
    
    for node in nodes:
        x, y = node.position.x, node.position.y
        
        if node.source == 1:
            plt.plot(x, y, 'o', color='blue', markersize=15,
                    markeredgecolor='black', markeredgewidth=2)
            plt.text(x, y, f'{node.node_id}',
                    fontsize=10, ha='center', va='center', color='white')
        elif node.node_id in offices:
            plt.plot(x, y, 's', color='orange', markersize=12,
                    markeredgecolor='black', markeredgewidth=2)
            plt.text(x + 200, y, str(node.node_id), fontsize=8, ha='left', va='center')
        else:
            plt.plot(x, y, 'o', color='green', markersize=8,
                    markeredgecolor='black', markeredgewidth=1)
            plt.text(x + 150, y, str(node.node_id), fontsize=7, ha='left', va='center')
    
    for conn in new_connections:
        new_node = next(n for n in nodes if n.node_id == conn['new_node_id'])
        x, y = new_node.position.x, new_node.position.y
        
        plt.plot(x, y, '*', color='red', markersize=20,
                markeredgecolor='black', markeredgewidth=2)
    
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(f'Red de Agua con nuevos nodos {title_suffix}')
    plt.grid(alpha=0.3)
    if new_connections:
        plt.legend()
    plt.tight_layout()
    saveFigure(output_path)


def main():
    files = ["FOS.txt", "HAN.txt", "NYT.txt", "PES.txt"]
    
    # Crear carpeta principal de resultados
    results_dir = "resultados"
    os.makedirs(results_dir, exist_ok=True)
    
    for file in files:
        instance_name = os.path.splitext(file)[0]
        print(f"Instancia actual: {file}")
        print()
        
        # Crear carpeta para esta instancia
        instance_dir = os.path.join(results_dir, instance_name)
        before_dir = os.path.join(instance_dir, "antes")
        after_dir = os.path.join(instance_dir, "despues")
        
        os.makedirs(before_dir, exist_ok=True)
        os.makedirs(after_dir, exist_ok=True)
        
        # Parte 1 - Lectura de instancias
        nodes, pipes, offices, new_nodes, sources = getData(file)
        
        # Antes de la expansión
        
        # Parte 2 - Longitudes de tuberias
        pipeLengthsWithSave(nodes, pipes, offices, new_nodes, 
                           os.path.join(before_dir, "01_longitudes_tuberias.png"), 
                           show_new_nodes=False)

        # Parte 3 - Sectorizacion
        node_to_sector_before, closed_pipes_before, open_pipes_before, source_distances_before = sectorize(nodes, pipes, sources)
        visualizeSectorizationWithSave(nodes, offices, new_nodes, node_to_sector_before, 
                                      closed_pipes_before, open_pipes_before, sources,
                                      os.path.join(before_dir, "02_sectorizacion.png"),
                                      show_new_nodes=False)

        # Parte 4 - Frescura del agua
        sector_freshness_before = freshnessAnalysis(nodes, node_to_sector_before, sources, source_distances_before)
        freshnessReport(sector_freshness_before)
        
        # Parte 5 - Flujo máximo
        max_flow_results_before = maxFlowAnalysis(nodes, pipes, node_to_sector_before, 
                                          sources, sector_freshness_before)
        maxFlowReport(max_flow_results_before)
        visualizeMaxFlowWithSave(nodes, pipes, node_to_sector_before, offices, new_nodes,
                                max_flow_results_before, sources,
                                os.path.join(before_dir, "03_flujo_maximo.png"),
                                show_new_nodes=False)
        
        # Parte 6 - Muestras de calidad del agua
        sampling_results_before = waterSamplingAnalysis(nodes, pipes, offices)
        samplingReport(sampling_results_before, nodes, pipes)
        visualizeSamplingWithSave(nodes, pipes, offices, new_nodes, sampling_results_before,
                                 os.path.join(before_dir, "04_muestreo"),
                                 show_new_nodes=False)
        
        # Generar reporte
        report_before = generateReport(instance_name, nodes, pipes, offices, node_to_sector_before,
                                      closed_pipes_before, sector_freshness_before,
                                      max_flow_results_before, sampling_results_before, sources,
                                      is_after=False)
        saveReport(os.path.join(before_dir, "reporte.txt"), instance_name, report_before)
        
        print()

        # Después de la expansión
        
        print("Red de distribución expandida")
        
        nodes_updated, pipes_updated, new_connections = connectNewNodes(nodes, pipes, new_nodes)
        newNodesReport(new_connections)
        
        # Visualizar red con nuevos nodos
        visualizeNewConnectionsWithSave(nodes_updated, pipes_updated, offices, new_connections,
                                       os.path.join(after_dir, "00_nuevas_conexiones.png"),
                                       "(DESPUÉS)")
        
        # Parte 2 - Longitudes de tuberías
        pipeLengthsWithSave(nodes_updated, pipes_updated, offices, [],
                           os.path.join(after_dir, "01_longitudes_tuberias.png"),
                           show_new_nodes=False)
        
        # Parte 3 - Sectorizacion
        node_to_sector_after, closed_pipes_after, open_pipes_after, source_distances_after = sectorize(nodes_updated, pipes_updated, sources)
        visualizeSectorizationWithSave(nodes_updated, offices, [], node_to_sector_after,
                                      closed_pipes_after, open_pipes_after, sources,
                                      os.path.join(after_dir, "02_sectorizacion.png"),
                                      show_new_nodes=False)

        # Parte 4 - Frescura del agua
        sector_freshness_after = freshnessAnalysis(nodes_updated, node_to_sector_after, sources, source_distances_after)
        freshnessReport(sector_freshness_after)
        
        # Parte 5 - Flujo máximo
        max_flow_results_after = maxFlowAnalysis(nodes_updated, pipes_updated, node_to_sector_after, 
                                          sources, sector_freshness_after)
        maxFlowReport(max_flow_results_after)
        visualizeMaxFlowWithSave(nodes_updated, pipes_updated, node_to_sector_after, offices, [],
                                max_flow_results_after, sources,
                                os.path.join(after_dir, "03_flujo_maximo.png"),
                                show_new_nodes=False)
        
        # Parte 6 - Muestras de calidad del agua
        sampling_results_after = waterSamplingAnalysis(nodes_updated, pipes_updated, offices)
        samplingReport(sampling_results_after, nodes_updated, pipes_updated)
        visualizeSamplingWithSave(nodes_updated, pipes_updated, offices, [], sampling_results_after,
                                 os.path.join(after_dir, "04_muestreo"),
                                 show_new_nodes=False)
        
        # Generar reporte
        report_after = generateReport(instance_name, nodes_updated, pipes_updated, offices,
                                     node_to_sector_after, closed_pipes_after,
                                     sector_freshness_after, max_flow_results_after,
                                     sampling_results_after, sources, is_after=True)
        saveReport(os.path.join(after_dir, "reporte.txt"), instance_name, report_after)
        
        print(f"\nResultados guardados en: {instance_dir}")
        print(f"  - Carpeta 'antes': {before_dir}")
        print(f"  - Carpeta 'despues': {after_dir}")


if __name__ == '__main__':
    main()