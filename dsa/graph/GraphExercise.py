""" 
DFS vs BFS

1. Depth-First Search (DFS)
Example: Solving a Maze or Puzzle
Problem: Given a maze, find a path from the start to the goal.
Why DFS:
DFS explores one path deeply before backtracking, which is suitable for finding a single feasible solution.
It is memory-efficient, as it only needs to keep track of the current path and backtracking history.
How DFS Helps:
DFS can be used to traverse the maze until it either reaches the goal or determines that the path is a dead end, at which point it backtracks to explore other possible routes.
Visualization:
Imagine solving a maze where you keep walking down one corridor until you hit a wall, then backtrack to try a different path. DFS mirrors this approach.

2. Breadth-First Search (BFS)
Example: Finding the Shortest Path in an Unweighted Graph
Problem: In a social network, find the shortest path (in terms of the number of connections) between two users.
Why BFS:
BFS explores all neighbors at the current level before moving deeper, guaranteeing the shortest path in terms of the number of edges.
How BFS Helps:
BFS starts from the source user and explores all directly connected users (friends). Then it moves to their friends' friends (next level), ensuring the shortest path is found first.
Visualization:
If you want to find the shortest way to connect to someone in a social network, BFS helps explore all possible first-degree connections before moving to second-degree connections.

Key Takeaways:
DFS is ideal for exploring all possible paths or solving puzzles where memory efficiency is important.
BFS is best for finding the shortest path in unweighted graphs or when level-order traversal is required.
 """

""" 
Topological sort, DFS
Sorts the given actions in such a way that if there is a dependency of one action on another, 
then dependent action always comes later than it parent action.
O(v+e)/v(v+e)
 """
def topological_sort(graph):

    def topological_sort_util(vertex, visited, stack):
        global graph
        if vertex not in visited:
            visited.add(vertex)
            edges= graph.get(vertex, [])
            for edge in edges:
                if edge not in visited:
                    topological_sort_util(edge, visited, stack)    
            stack.append(vertex)

    visited= set()
    stack=[]
    for vertex in graph:
        if vertex not in visited:
            topological_sort_util(vertex, visited, stack)

    stack.reverse()
    print(stack)

from collections import deque
""" 
Single source shortest path using BFS
Only for unweighted (directed + undirected), easier and better performance than dijkstra and bellman algos 
Note: BFS method works only for unweighted graphs because for weighted graph breadth way could not be best paths
      DFS cannot be used for SSSP because it goes till the end and then backtrack. eg for this graph A->B->C A->C DFS will give first path for A to C
O(v+e)/O(v+e)
 """
def single_source_shortest_path_bfs(graph:dict, source_vertex:str):
    visited= set()
    my_queue= deque()
    path_dict= {source_vertex: [source_vertex]}

    my_queue.append(source_vertex)

    while my_queue:
        vertex= my_queue.popleft()
        if vertex not in visited:
            visited.add(vertex)
            adj_vertices= graph.get(vertex, [])
            for adj_vertex in adj_vertices:
                if adj_vertex not in visited:
                    my_queue.append(adj_vertex)
                    parent_vertex_path= path_dict[vertex]
                    path_dict[adj_vertex]=parent_vertex_path+[adj_vertex]

    print(path_dict)

""" 
Dijkstra's algorithm
Mainly used for weighted graphs without negative cycle, easier and better than bellman 
O(v square)/O(v).
In this algo we add total min weight of target from soure to min heap, where as in prim's algo we add pair weights to min heap
 """
import heapq
class Edge:
    def __init__(self, weight, target_vertex):
        self.weight=weight
        self.target_vertex= target_vertex

class Vertex:
    def __init__(self, name):
        self.name= name
        self.min_weight= float("inf")
        self.edges= []
        self.predecesor= None
        self.visited= False

    def __lt__(self, other_vertex):
        return self.min_weight< other_vertex.min_weight
        
    def add_edge(self, weight, target_vertex):
        edge= Edge(weight, target_vertex)
        self.edges.append(edge)

class Dijkstra:
    def __init__(self):
        self.min_wieght_vertex_heap=[]
        
    def calculate(self, start_vertex:Vertex):
        start_vertex.min_weight=0
        heapq.heappush(self.min_wieght_vertex_heap, start_vertex)

        while self.min_wieght_vertex_heap:
            vertex= heapq.heappop(self.min_wieght_vertex_heap)

            if not vertex.visited:
                vertex.visited=True
                for edge in vertex.edges:
                    target_vertex_weight= vertex.min_weight+edge.weight
                    if target_vertex_weight< edge.target_vertex.min_weight:
                        edge.target_vertex.min_weight= target_vertex_weight
                        edge.target_vertex.predecesor= vertex
                        heapq.heappush(self.min_wieght_vertex_heap, edge.target_vertex)

    def get_sortest_path(self, target_vertex):
        print(f'sortest path to {target_vertex.name} from source: {target_vertex.min_weight}')
        while target_vertex:
            print(target_vertex.name, end=" ")
            target_vertex=target_vertex.predecesor

""" 
Bellman Ford algo
Mainly used to tackle negative cycle, use only for this purpose as its costlier than Dijkstra algo
O(ve)/O(v)
Follow for comparison: https://www.udemy.com/course/data-structures-and-algorithms-bootcamp-in-python/learn/lecture/22158226#overview
 """
class Node:
    def __init__(self, value):
        self.value= value
        self.predecesor= None

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other_node):
        if isinstance(Node, other_node):
            return self.value==other_node.value
        return NotImplemented

class Graph:
    def __init__(self, num_vertices):
        self.num_vertices= num_vertices
        self.nodes=[]
        self.edges=[]
    
    def add_node(self, node):
        self.nodes.append(node)

    def add_edge(self, s, d, w):
        self.edges.append([s, d, w])

    def print_solution(self, dict):
        print("vertex distance from source")
        for k, v in dict.items():
            print(f'{k.value} : {v}')
            vertex=k
            while vertex:
                print(vertex.value, end=' ')
                vertex= vertex.predecesor
            print()


    def bellManFord(self, source):
        dict= {i: float('inf') for i in self.nodes}
        dict[source]= 0

        for _ in range(self.num_vertices-1):
            for s, d, w in self.edges:
                if dict[s]!=float('inf') and dict[d] > dict[s] + w:
                    d.predecesor= s
                    dict[d]= dict[s] + w


        for s, d, w in self.edges:
            if dict[d] > dict[s] + w:
                print("negative loop found")
                return
            
        self.print_solution(dict)

""" 
All pair sortest path:
Consider each and every vertex as source and run any single source shortest path algo for each vertex as source
hence any algo will run v number of times.
Following are other algos for all pair sortest path.
 """

""" 
Floyd Warshall algo
We cannnot detect negative cycle with this algo because
to go through cycle we need to go via negative cycle participating vertex at least twice but this algo never runs loop twice via same vertex.
O(v cube)/O(v square)
compare all algos for all pair: https://www.udemy.com/course/data-structures-and-algorithms-bootcamp-in-python/learn/lecture/22208230#overview
 """

INF=9999
def print_solution(nV, graph):
    for i in range(nV):
        for j in range(nV):
            if graph[i][j]==INF:
                print('INF', end=" ")
            else:
                print(graph[i][j], end=" ")
        print("")

def floydAlgo(nV, graph):
    for k in range(nV):
        for i in range(nV):
            for j in range(nV):
                graph[i][j]= min(graph[i][j], graph[i][k]+graph[k][j])

    print_solution(nV, graph)

""" 
Minimum Spaning Tree
Cheapest way to connct all the vertices, its different from SSSP problem as here there is no single source
 """

""" 
Disjoint Set using union by rank and union by size
O(v)/O(v)
 """
class DisjointSet:
    def __init__(self, vertices):
        self.verties=vertices
        self.parent={i:i for i in vertices}
        self.rank=dict.fromkeys(vertices, 0)
        # or
        self.size=dict.fromkeys(vertices, 1)

    def find(self, vertex):
        if self.parent[vertex]==vertex:
            return vertex
        else:
            return self.find(self.parent[vertex])
        
    def unionByRank(self, vertex_x, vertex_y):
        ultimate_parent_of_x= self.find(vertex_x)
        ultimate_parent_of_y= self.find(vertex_y)

        if self.rank[ultimate_parent_of_x]>self.rank[ultimate_parent_of_y]:
            self.parent[ultimate_parent_of_y]= ultimate_parent_of_x
        elif self.rank[ultimate_parent_of_y]>self.rank[ultimate_parent_of_x]:
            self.parent[ultimate_parent_of_x]= ultimate_parent_of_y
        else:
            self.parent[ultimate_parent_of_y]=ultimate_parent_of_x
            self.rank[ultimate_parent_of_x]+=1

    # or
    def unionBySize(self, vertex_x, vertex_y):
        ultimate_parent_of_x= self.find(vertex_x)
        ultimate_parent_of_y= self.find(vertex_y)

        if self.size[ultimate_parent_of_y]>self.size[ultimate_parent_of_x]:
            self.parent[ultimate_parent_of_x]= ultimate_parent_of_y
            self.size[ultimate_parent_of_y]+=self.size[ultimate_parent_of_x]
        else:
            self.parent[ultimate_parent_of_y]= ultimate_parent_of_x
            self.size[ultimate_parent_of_x]+=self.size[ultimate_parent_of_y]


""" 
Kruskal algo to find MST
 """
class Graph_K:
    def __init__(self, nV):
        self.nV= nV
        self.nodes=[]
        self.edges=[]
        self.MST=[]

    def add_nodes(self, val):
        self.nodes.append(val)

    def add_edge(self, s, d, w):
        self.edges.append([s, d, w])

    def print_solution(self):
        for s,d,w in self.MST:
            print(f'{s}-{d}:{w}')

    def kruskal(self):
        self.edges= sorted(self.edges, key= lambda edge: edge[2])
        ds= DisjointSet(self.nodes)
        for edge in self.edges:
            s= edge[0]
            d= edge[1]
            w= edge[2]

            ultimate_parent_of_s= ds.find(s)
            ultimate_parent_of_d= ds.find(d)

            if ultimate_parent_of_s != ultimate_parent_of_d:
                self.MST.append([s, d, w])
                ds.unionByRank(s, d)

            if len(self.MST)==self.nV-1:
                break

        self.print_solution()


""" 
Prim's algo to find MST
O(elogv)/O(e)
 """
class Graph_p:
    def __init__(self):
        self.nodes=[]

    def add_node(self, node):
        self.nodes.append(node)

class Node_p:
    def __init__(self, val):
        self.val=val
        self.edges=[]

    def add_edge(self, target_vertex, wt):
        edge=Edge_p(target_vertex, wt)
        self.edges.append(edge)
        
class Edge_p:
    def __init__(self, target_vertex:Node_p, wt):
        self.target_vertex=target_vertex
        self.wt=wt

class PrimsAlgo:
    def __init__(self):
        self.mst=[]
        self.visited=set()
        self.weight_min_heap= []
        self.weight_sum=0

    def solution(self, graph:Graph_p):
        first_node=graph.nodes[0]
        heap_tuple= (0, first_node, None)  #(edge_wt, node, parent)
        heapq.heappush(self.weight_min_heap, heap_tuple)

        while self.weight_min_heap:
            edge_wt, node, parent = heapq.heappop(self.weight_min_heap)
            if node.val not in self.visited:
                self.visited.append(node.val)
                if parent:
                    self.mst.append((parent.val, node.val))
                    self.weight_sum+=edge_wt
                for edge in node.edges:
                    target_node= edge.target_vertex
                    if target_node.val not in self.visited:
                        heap_tuple= (edge.wt, target_node, node)
                        heapq.heappush(self.weight_min_heap, heap_tuple)

        print(self.mst)
        print(self.weight_sum)

""" 
Kruskal vs Prim's

When to Choose?
Scenario	                            Preferred Algorithm
--------------                          -------------------
Sparse graph	                        Kruskal
Dense graph	Prim
Adjacency matrix representation	        Prim
Adjacency list representation	        Both work, but Prim is better for dense graphs
Disconnected graph (need a forest)	    Kruskal
Dynamic graph	                        Prim
Simpler implementation	                Kruskal
 """

""" 
Detect a cycle in directed graph using DFS
https://www.youtube.com/watch?v=9twcmtQj4DU&t=125s&ab_channel=takeUforward
 """
def detectCycleDirectedGraph(graph):

    def detectCycleUtil(graph, node, visited, path_visited):
        visited.append(node)
        path_visited.append(node)

        for adj in graph[node]:
            if adj not in visited:
                return detectCycleUtil(graph, adj, visited, path_visited)
            elif adj in path_visited:
                    return True
            
        path_visited.remove(node)
        return False
                    
    visited= []
    path_visited= []
    nodes= graph.keys()
    for n in nodes:
        if n not in visited:
            return detectCycleUtil(graph, n, visited, path_visited)
        
        return False


""" 
Detect cycle in undirected graph using DFS
https://www.youtube.com/watch?v=zQ3zgFypzX4&t=698s&ab_channel=takeUforward
 """
def detectCycleUndirecteGraph(graph):

    def detectCycleUtil(graph, node, visited, parent):
        visited.append(node)
        for adj_node in graph[node]:
            if adj_node not in visited:
                return detectCycleUtil(graph, adj_node, visited, node)
            elif adj_node!=parent:
                return True
        
        return False

    visited=[]
    nodes= graph.keys()
    for n in nodes:
        if n not in visited:
            return detectCycleUtil(graph, n, visited, -1)

    return False

""" 
Find if there is a path between two vertices in a directed graph
 """
""" 
Using DFS
 """
def findPathUsingDFS(graph, u, v):
    def findPathUtil(graph, node):
        if node==v:
            return True
        
        visited.add(node)

        for neighbour in graph[node]:
            if neighbour not in visited:
                return findPathUtil(graph, neighbour)
            
        return False
        
    visited=set()
    return findPathUtil(graph, u)

""" 
Using BFS
 """
def findPathUsingBFS(graph, u, v):
    my_queue= deque()
    visited=set()
    my_queue.append(u)

    while my_queue:
        node= my_queue.popleft()
        if node==v:
            return True
        visited.add(node)
        for neighbour in graph[node]:
            if neighbour not in visited:
                my_queue.append(neighbour)

    return False

""" 
Transitive closure of a graph using Floyd Warshall Algorithm.
Given a directed graph, determine if a vertex j is reachable from another vertex i for all vertex pairs (i, j) in the given graph. 
Here reachable means that there is a path from vertex i to j. The reach-ability matrix is called the transitive closure of a graph.
 """
class TransitiveClosure:
    def __init__(self, nV):
        self.nV= nV

    def print_solution(self, reach):
        print("Following matrix transitive closure of the given graph ")
        for i in range(self.nV):
            for j in range(self.nV):
                print(reach[i][j], end=" ")
            print("")

    def trasitive_closure(self, graph):
        reach= [i[:] for i in graph]

        for i in range(self.nV):
            for j in range(self.nV):
                for k in range(self.nV):
                    reach[i][j]= reach[i][j] or (reach[i][k] and reach[k][j])

        self.print_solution(reach)


""" 
Strongly Connected Components Using Kosaraju's Algorithm DFS
O(V+E)/O(V)
 """
from collections import defaultdict

class StronglyConnectedComponents:
    def __init__(self, vertices):
        """
        Initialize the graph with the given number of vertices.
        """
        self.graph = defaultdict(list)
        self.vertices = vertices

    def add_edge(self, u, v):
        """
        Add a directed edge from vertex u to vertex v.
        """
        self.graph[u].append(v)

    def _dfs(self, node, visited, stack):
        """
        Perform DFS and push nodes onto the stack based on their finish time.
        """
        visited.add(node)
        for neighbor in self.graph[node]:
            if neighbor not in visited:
                self._dfs(neighbor, visited, stack)
        stack.append(node)

    def _dfs_transposed(self, node, visited, component, transposed_graph):
        """
        Perform DFS on the transposed graph to find SCCs.
        """
        visited.add(node)
        component.append(node)
        for neighbor in transposed_graph[node]:
            if neighbor not in visited:
                self._dfs_transposed(neighbor, visited, component, transposed_graph)

    def find_sccs(self):
        """
        Find all strongly connected components in the graph using Kosaraju's Algorithm.
        """
        # Step 1: Perform DFS and record the finishing order in a stack
        visited = set()
        stack = []
        for vertex in range(self.vertices):
            if vertex not in visited:
                self._dfs(vertex, visited, stack)

        # Step 2: Transpose the graph (reverse all edges)
        transposed_graph = defaultdict(list)
        for u in self.graph:
            for v in self.graph[u]:
                transposed_graph[v].append(u)

        # Step 3: Perform DFS on the transposed graph in the order of the stack
        visited.clear()
        sccs = []
        while stack:
            node = stack.pop()
            if node not in visited:
                component = []
                self._dfs_transposed(node, visited, component, transposed_graph)
                sccs.append(component)

        return sccs

""" 
Check if a graph is strongly connected Using Kosaraju's Algorithm DFS
O(V+E)
 """
from collections import defaultdict

class StronglyConnectedGraphChecker:
    def __init__(self, vertices):
        self.graph = defaultdict(list)
        self.vertices = vertices

    def add_edge(self, u, v):
        """
        Add a directed edge from u to v.
        """
        self.graph[u].append(v)

    def _dfs(self, node, visited):
        """
        Perform DFS starting from the given node.
        """
        visited.add(node)
        for neighbor in self.graph[node]:
            if neighbor not in visited:
                self._dfs(neighbor, visited)

    def _transpose_graph(self):
        """
        Transpose the graph by reversing all edges.
        """
        transposed_graph = defaultdict(list)
        for node in self.graph:
            for neighbor in self.graph[node]:
                transposed_graph[neighbor].append(node)
        return transposed_graph

    def is_strongly_connected(self):
        """
        Check if the graph is strongly connected.
        """
        # Step 1: Perform DFS from any vertex (e.g., 0)
        visited = set()
        self._dfs(0, visited)

        # Check if all vertices were visited
        if len(visited) != self.vertices:
            return False

        # Step 2: Transpose the graph
        transposed_graph = self._transpose_graph()

        # Step 3: Perform DFS on the transposed graph
        visited = set()
        self.graph = transposed_graph  # Temporarily replace with transposed graph
        self._dfs(0, visited)

        # Check if all vertices were visited in the transposed graph
        return len(visited) == self.vertices

# Kosaraju's Algorithm
# if __name__ == "__main__":
#     scc_finder = StronglyConnectedComponents(vertices=8)
#     scc_finder.add_edge(0, 1)
#     scc_finder.add_edge(1, 2)
#     scc_finder.add_edge(2, 0)
#     scc_finder.add_edge(1, 3)
#     scc_finder.add_edge(3, 4)
#     scc_finder.add_edge(4, 5)
#     scc_finder.add_edge(5, 3)
#     scc_finder.add_edge(6, 5)
#     scc_finder.add_edge(6, 7)

#     sccs = scc_finder.find_sccs()
#     print("Strongly Connected Components:", sccs)


#Transitive Closure
# tc= TransitiveClosure(4)
 
# graph = [[1, 1, 0, 1],
#          [0, 1, 1, 0],
#          [0, 0, 1, 1],
#          [0, 0, 0, 1]]
 
# #Print the solution
# tc.trasitive_closure(graph)

# is a path between two vertices in a directed graph
# graph = {
#     "A": ["B", "C"],
#     "B": ["D"],
#     "C": ["E"],
#     "D": ["F"],
#     "E": [],
#     "F": []
# }

# source = "A"
# target = "F"

# # print(findPathUsingDFS(graph,source,target))
# print(findPathUsingBFS(graph,source,target))

#cycle in undirected graph through DFS
# nV=4
# grapph= {i:[] for i in range(nV)}
# grapph[1].append(0)
# grapph[0].append(1)
# # grapph[0].append(2)
# # grapph[2].append(0)
# grapph[1].append(2)
# grapph[2].append(1)

# print(detectCycleUndirecteGraph(grapph))

#cycle in directed graph through DFS
# nV=4
# grapph= {i:[] for i in range(nV)}
# grapph[0].append(1)
# grapph[0].append(2)
# grapph[1].append(2)
# grapph[2].append(0)
# grapph[2].append(3)
# grapph[3].append(3)

# print(detectCycleDirectedGraph(grapph))


# graph= {'A':['C'],
#         'C':['E'],
#         'E':['H', 'F'],
#         'F':['G'], 
#         'B':['D', 'C'],
#         'D':['F']
#         }

# topological_sort(graph)

# graph= {'A':['C', 'B'],
#         'C':['A', 'E', 'D'],
#         'B':['A', 'D', 'G'],
#         'E':['C', 'F'], 
#         'D':['B', 'C', 'F'],
#         'G':['B', 'F'],
#         'F':['E', 'D', 'G']
#         }

# single_source_shortest_path_bfs(graph, 'A')

# vertexA= Vertex('A')
# vertexB= Vertex('B')
# vertexC= Vertex('C')
# vertexD= Vertex('D')
# vertexE= Vertex('E')
# vertexF= Vertex('F')
# vertexG= Vertex('G')
# vertexH= Vertex('H')

# vertexA.add_edge(6, vertexB)
# vertexA.add_edge(10, vertexC)
# vertexA.add_edge(9, vertexD)

# vertexB.add_edge(5, vertexD)
# vertexB.add_edge(16, vertexE)
# vertexB.add_edge(13, vertexF)

# vertexC.add_edge(6, vertexD)
# vertexC.add_edge(5, vertexH)
# vertexC.add_edge(21, vertexG)

# vertexD.add_edge(8, vertexF)
# vertexD.add_edge(7, vertexH)

# vertexE.add_edge(10, vertexG)

# vertexF.add_edge(4, vertexE)
# vertexF.add_edge(12, vertexG)

# vertexH.add_edge(2, vertexF)
# vertexH.add_edge(14, vertexG)

# dijkstra= Dijkstra()

# dijkstra.calculate(vertexA)
# dijkstra.get_sortest_path(vertexG)

# graph= Graph(5)

# nodeA= Node('A')
# nodeB= Node('B')
# nodeC= Node('C')
# nodeD= Node('D')
# nodeE= Node('E')

# graph.add_node(nodeA)
# graph.add_node(nodeB)
# graph.add_node(nodeC)
# graph.add_node(nodeD)
# graph.add_node(nodeE)

# graph.add_edge(nodeA, nodeB, 6)
# # graph.add_edge('A', 'D', -6)
# graph.add_edge(nodeB, nodeD, 1)
# graph.add_edge(nodeC, nodeA, 3)
# graph.add_edge(nodeD, nodeB, 1)
# graph.add_edge(nodeD, nodeC, 1)
# graph.add_edge(nodeE, nodeC, 4)
# graph.add_edge(nodeE, nodeD, 2)

# graph.bellManFord(nodeE)


# graph=[[0, 8, 1, INF],
#        [INF, 0, INF, 1],
#        [INF, 2, 0, 9],
#        [4, INF, INF, 0]
#        ]

# floydAlgo(4, graph)

# vertices= [1, 2, 3, 4, 5, 6, 7]

# disSet= DisjointSet(vertices)
# print(disSet.parent)
# print(disSet.rank)

# # disSet.unionByRank(1,2)
# # disSet.unionByRank(2,3)
# # disSet.unionByRank(4,5)
# # disSet.unionByRank(6,7)
# # disSet.unionByRank(5,6)
# # disSet.unionByRank(3,7)

# disSet.unionBySize(1,2)
# disSet.unionBySize(2,3)
# disSet.unionBySize(4,5)
# disSet.unionBySize(6,7)
# disSet.unionBySize(5,6)
# disSet.unionBySize(3,7)

# print(disSet.parent)

# g= Graph_K(5)

# g.add_nodes('A')
# g.add_nodes('B')
# g.add_nodes('C')
# g.add_nodes('D')
# g.add_nodes('E')

# g.add_edge('A', 'B', 5)
# g.add_edge('A', 'C', 13)
# g.add_edge('A', 'E', 15)
# g.add_edge('B', 'A', 5)
# g.add_edge('B', 'C', 10)
# g.add_edge('B', 'D', 8)
# g.add_edge('C', 'A', 13)
# g.add_edge('C', 'B', 10)
# g.add_edge('C', 'E', 20)
# g.add_edge('D', 'C', 6)

# g.kruskal()

# g= Graph_p()
# node0= Node_p(0)
# node1= Node_p(1)
# node2= Node_p(2)
# node3= Node_p(3)
# node4= Node_p(4)

# node0.add_edge(node1, 2)
# node0.add_edge(node3, 6)
# node1.add_edge(node0, 2)
# node1.add_edge(node2, 3)
# node1.add_edge(node3, 8)
# node1.add_edge(node4, 5)
# node2.add_edge(node1, 3)
# node2.add_edge(node4, 7)
# node3.add_edge(node0, 6)
# node3.add_edge(node1, 8)
# node4.add_edge(node1, 5)
# node4.add_edge(node2, 7)

# g.add_node(node0)
# g.add_node(node1)
# g.add_node(node2)
# g.add_node(node3)
# g.add_node(node4)

# pa= PrimsAlgo()
# pa.solution(g)
