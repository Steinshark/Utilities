## build a graph 


class Edge:
    def __init__(self,nodeA,nodeB,weight=None):
        self.nodeA = nodeA
        self.nodeB = nodeB
        self.weight = weight
class AdjacencyMatrix:
    def __init__(self,nodesList,isDirected,isWeighted):
        self.isDirected = isDirected
        self.isWeighted = isWeighted
        self.adjacencyMatrix = {node : {node : None for node in nodesList} for node in nodesList}

    def connect(self,nodeA,nodeB,weight=None):
        # Weighted graph 
        if self.isWeighted:
            if weight is None:
                print(f'Adding {nodeA} -> {nodeB} failed. No weight specified for weighted graph.')
            else:
                self.adjacencyMatrix[nodeA][nodeB] = weight 
            
            if not self.isDirected:
                self.adjacencyMatrix[nodeB][nodeA] = weight
        # Unweighted graph
        else:
            self.adjacencyMatrix[nodeA][nodeB] = True
            if not self.isDirected:
                self.adjacencyMatrix[nodeB][nodeA] = True

    def isConnected(self,nodeA,nodeB):
        return not self.adjacencyMatrix[nodeA][nodeB] is None
    
    def getNeighbors(self,nodeA):
        return self.adjacencyMatrix[nodeA]
    
    def getConnection(self,nodeA,nodeB):
        if not isConnected(nodeA,nodeB):
            print(f'Graph does not hold {nodeA}->{nodeB}')
            return 
        else:
            return self.adjacencyMatrix[nodeA][nodeB]

     
class Graph:
    def __init__(self,nodesList,isDirected,isWeighted):
        self.nodes = nodesList
        self.connections = AdjacencyMatrix(nodesList,isDirected,isWeighted)
    def connect(self,nodeA,nodeB,weight=None):
        return self.connections.connect(nodeA,nodeB,weight)
    def isConnected(self,nodeA,nodeB):
        return self.connections.isConnected(nodeA,nodeB)
    def getNeighbors(self,nodeA):
        return self.connections.getNeighbors(nodeA)
    def getConnection(nodeA,nodeB):
        return self.connections.getConection(nodeA,nodeB)

if __name__ == '__main__':
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    import random
    g = Graph(alphabet,False,False)
    for letter in alphabet:
        for letter2 in alphabet:
            if random.randint(0,4) % 3 == 0:
                g.connect(letter,letter2)

