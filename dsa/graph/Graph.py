from collections import deque
class Graph:
    def __init__(self):
        self.my_dict={}

    def add_vertex(self, vertex):
        if vertex not in self.my_dict.keys():
            self.my_dict[vertex]=[]
            return True
        return False
    
    def print_graph(self):
        for vertex in self.my_dict.keys():
            print(vertex, ':', self.my_dict[vertex])
    
    def add_edge(self, vertex1, vertex2):
        if vertex1 in self.my_dict.keys() and vertex2 in self.my_dict.keys(): 
            self.my_dict[vertex1].append(vertex2)
            self.my_dict[vertex2].append(vertex1)
            return True
        return False
    
    def remove_edge(self, vertex1, vertex2):
        if vertex1 in self.my_dict.keys() and vertex2 in self.my_dict.keys():
            try:
                self.my_dict[vertex1].remove(vertex2)
                self.my_dict[vertex2].remove(vertex1)
            except ValueError:
                pass
            return True
        return False
    
    def remove_vertex(self, vertex):
        if vertex in self.my_dict.keys():
            for connected_vertex in self.my_dict[vertex]:
                self.my_dict[connected_vertex].remove(vertex)

            del self.my_dict[vertex]
            return True
        return False

    """ 
     O(v+e)/O(v)
    """
    def bfs(self, start_vertex):
        visited= set()
        my_queue= deque()

        my_queue.append(start_vertex)
        while my_queue:
            vertex= my_queue.popleft()
            if vertex not in visited:
                visited.add(vertex)
                print(vertex)
                for adjacent_vertex in self.my_dict[vertex]:
                    if adjacent_vertex not in visited:
                        my_queue.append(adjacent_vertex)

    """ 
     O(v+e)/O(v)
    """
    def dfs(self, start_vertex):
        visited= set()
        my_stack= []

        my_stack.append(start_vertex)
        while my_stack:
            vertex= my_stack.pop()
            if vertex not in visited:
                visited.add(vertex)
                print(vertex)
                for adjacent_vertex in self.my_dict[vertex]:
                    if adjacent_vertex not in visited:
                        my_stack.append(adjacent_vertex)

myGraph= Graph()
myGraph.add_vertex('A')
myGraph.add_vertex('B')
myGraph.add_vertex('C')
myGraph.add_vertex('D')
myGraph.add_vertex('E')
myGraph.print_graph()

myGraph.add_edge('A', 'B')
myGraph.add_edge('A', 'C')
myGraph.add_edge('B', 'E')
myGraph.add_edge('E', 'D')
myGraph.add_edge('D', 'C')
myGraph.print_graph()

# myGraph.remove_edge('A', 'B')
# myGraph.print_graph()

myGraph.bfs('A')
print('---')
myGraph.dfs('A')