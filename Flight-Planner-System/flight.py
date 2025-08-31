class Flight:
    def __init__(self, flight_no, start_city, departure_time, end_city, arrival_time, fare):
        """ Class for the flights

        Args:
            flight_no (int): Unique ID of each flight
            start_city (int): The city no. where the flight starts
            departure_time (int): Time at which the flight starts
            end_city (int): The city no where the flight ends
            arrival_time (int): Time at which the flight ends
            fare (int): The cost of taking this flight
        """
        self.flight_no = flight_no
        self.start_city = start_city
        self.departure_time = departure_time
        self.end_city = end_city
        self.arrival_time = arrival_time
        self.fare = fare
        
"""
If there are n flights, and m cities:

1. Flight No. will be an integer in {0, 1, ... n-1}
2. Cities will be denoted by an integer in {0, 1, .... m-1}
3. Time is denoted by a non negative integer - we model time as going from t=0 to t=inf
"""

class Heap:
    '''
    Class to implement a heap with general comparison function
    '''

    def __init__(self, comparison_function, init_array = []):
        # same heap from assignment 3
        self.comparison_function = comparison_function
        # Initializing the heap with the given array
        self.heap = init_array  
        self.convertheap()

    def convertheap(self):
        #Heapifying the initial array for converting to heap
        for i in range(len(self.heap) // 2, -1, -1):
            self.downheap(i)

    def insert(self, value):
        
        #Insert a value into the heap and heap bigad jaye to usko sahi kar rahe hai
        
        self.heap.append(value)
        self.upheap(len(self.heap) - 1)

    def extract(self):
        '''
        Extracts the value from the top of heap
        '''
       
        if len(self.heap) == 0:
            return 
            
        top = self.heap[0]
        self.heap[0] = self.heap[-1]
        self.heap.pop()  

        if len(self.heap) > 0:
            self.downheap(0)
        return top

    def top(self):
         # to maintain heap property
        if len(self.heap) == 0:
            return None
        return self.heap[0]

    def upheap(self, j):
        # to maintain heap property
        parent = (j - 1) // 2  
        if j > 0 and self.comparison_function(self.heap[j], self.heap[parent]):
            self.heap[j], self.heap[parent] = self.heap[parent], self.heap[j]
            self.upheap(parent)

    def downheap(self, j):
       
        left = (2 * j + 1)
        right = (2 * j + 2)
        smallest = j
        if left < len(self.heap) and self.comparison_function(self.heap[left], self.heap[smallest]):
            smallest = left
        if right < len(self.heap) and self.comparison_function(self.heap[right], self.heap[smallest]):
            smallest = right
        if smallest != j:
            self.heap[j], self.heap[smallest] = self.heap[smallest], self.heap[j]
            self.downheap(smallest)

    def print_heap(self):
        if not self.heap:
            print("Heap is empty")
            return
        for idx in range(len(self.heap)):
            element = self.heap[idx]
            print(f"Index {idx}: ID = {element.id}, Size = {element.size}, Arrival Time = {element.arrival_time}")

    def is_empty(self):
        if len(self.heap)==0:
            return True
        else :
            return False
    def find(self, city):
        
        i = 0
        while i < len(self.heap):
            if self.heap[i][0] == city:
                return i
            i += 1
        return None
def comparator_function(a,b):
    return a[1]<b[1]
def comparator_function_min(a, b):
   
    if a[1] != b[1]:
        return a[1] < b[1]
  
    return a[2] < b[2]


class Node:
   
    def __init__(self, data):
        self.data = data
        self.next = None


class Queue:
    # Queue implementation .
    def __init__(self):
        self.front_node = None  
        self.rear_node = None  
        self.count = 0         
    def enqueue(self, item):
        """Add an item to the end of the queue."""
        nodetoinsert= Node(item)
        if self.is_empty():
            self.front_node = nodetoinsert
            self.rear_node = nodetoinsert
        else:
            self.rear_node.next = nodetoinsert
            self.rear_node = nodetoinsert
        self.count += 1

    def dequeue(self):
        if  self.is_empty():
            return None
        
        else: 
            top= self.front_node.data
            self.front_node = self.front_node.next
            if self.front_node is None:  
                self.rear_node = None
        self.count -= 1
        return top
         

    def is_empty(self):
        if self.front_node is None:
            return True
        else:
            return False

    

    
    def size(self):
        ans= self.count
        return ans

