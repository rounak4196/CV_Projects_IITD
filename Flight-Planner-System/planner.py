from flight import Flight , comparator_function ,comparator_function_min
from flight import Heap
from flight import Queue

class Planner:
    def __init__(self, flights):
        """The Planner

        Args:
            flights (List[Flight]): A list of information of all the flights (objects of class Flight)
        """
        self.flights = flights
    
        # Calculating the number of cities based on the maximum flight number
        totatcities = 0 # Initialize with a very small value

        for flight in flights:
            if flight.start_city > totatcities:
                totatcities = flight.start_city
            if flight.end_city > totatcities:
                totatcities = flight.end_city

        self.noofcities = totatcities + 1  # Total number of cities


        # Initializing adjacency list for departures
        self.cities = [[] for c in range(self.noofcities)]
        self.departure_adj = [[] for _ in range(self.noofcities)]
        for flight in flights:
            self.departure_adj[flight.start_city].append(flight)


        for flight in flights:
            self.cities[flight.start_city].append(flight)
        self.flightadj=[(flight,[]) for flight in flights]
        i = 0  # Initialize the i for iteration
        while i < len(self.flights):  
            current_flight = self.flights[i]  # Fetch the current flight
            destination_city = current_flight.end_city  # Get the destination city of the flight

            # Iterate over flights starting from the destination city
            for next_flight in self.cities[destination_city]:
                # Check if the next flight departs at least 20 minutes after the current flight arrives
                if next_flight.departure_time >= current_flight.arrival_time + 20:
                    # Add the next flight to the adjacency list for the current flight
                    self.flightadj[i][1].append(next_flight)
            
            i += 1  # Move to the next flight


    
    def least_flights_earliest_route(self, start_city, end_city, t1, t2):
        """
        Return List[Flight]: A route from start_city to end_city, which departs after t1 (>= t1) and
        arrives before t2 (<=) satisfying:
        - The route has the least number of flights
        - Within routes with same number of flights, arrives the earliest
        """
        if start_city == end_city:
            return []
            
        path = []
        min_flights = float('inf')
        earliest_arrival = float('inf')
        q = Queue()
        parent = [-1] * len(self.flights)
        
        # Track minimum flights and earliest arrival time to reach each flight
        min_no_flight = [(float('inf'), float('inf'))] * len(self.flights)  # (num_flights, arrival_time)
        
        # Try each possible starting flight from start_city
        for starting_flight in self.cities[start_city]:
            if starting_flight.departure_time < t1:
                continue
                
            # Initialize for BFS using queue
            q.enqueue((starting_flight, 1))  # (flight, num_flights)
            min_no_flight[starting_flight.flight_no] = (1, starting_flight.arrival_time)
            
        while not q.is_empty():
            current = q.dequeue()
            if current is None:
                break
                
            current_flight, currentflight_totalcount = current
            
            # If  reached end_city, checking if this is the best route
            if current_flight.end_city == end_city and current_flight.arrival_time <= t2:
                if (currentflight_totalcount < min_flights or (currentflight_totalcount == min_flights and current_flight.arrival_time < earliest_arrival)):
                    min_flights = currentflight_totalcount
                    earliest_arrival = current_flight.arrival_time
                    
                    
                    path = []
                    currentflight_number = current_flight.flight_no
                    # bactraking upto parent to final ans
                    while currentflight_number != -1:
                        path.append(self.flights[currentflight_number])
                        currentflight_number = parent[currentflight_number]
                    path = path[::-1]

                    
                continue
            
            # finding all possible next flights
            for next_flight in self.flightadj[current_flight.flight_no][1]:
                if next_flight.arrival_time <= t2:
                    new_flight_count = currentflight_totalcount + 1
                    
                    # Only consider this path if it has fewer flights or same flights but earlier arrival
                    curr_min_flights, curr_arrival_time = min_no_flight[next_flight.flight_no]
                    if (new_flight_count < curr_min_flights ):
                        min_no_flight[next_flight.flight_no] = (new_flight_count, next_flight.arrival_time)
                        parent[next_flight.flight_no] = current_flight.flight_no
                        q.enqueue((next_flight, new_flight_count))
                    elif (new_flight_count == curr_min_flights and next_flight.arrival_time < curr_arrival_time): 
                        min_no_flight[next_flight.flight_no] = (new_flight_count, next_flight.arrival_time)
                        parent[next_flight.flight_no] = current_flight.flight_no
                        q.enqueue((next_flight, new_flight_count))  
        return path



            

       

        # def bfs(node , end_city , visited):
            
        #     q.push(node)
        
        # INF = 10**9 
        # max_ans  =  INF 

        # for i in range(self.cities[start_city]):
        #     visited = set() 
        #     ans  = bfs(self.cities[i] , end_city , visited ) 

        #     if ans < max_ans :
        #         max_ans = ans 
        #         store = i 
        pass
            

    
    def cheapest_route(self, start_city, end_city, t1, t2):
        """
        Return List[Flight]: A route from start_city to end_city, which:
        - Departs after t1 (>= t1)
        - Arrives before t2 (<= t2)
        - Is the cheapest possible route
        """



        # Handle same city case
        if start_city == end_city:
            return []
            
        final_ans = []
        cheapest_total_fare = float('inf')
        
        # Initialize Dijkstra's algorithm data structures
        pq = Heap(comparator_function)  # Min-heap based on fare
        min_fares = [float('inf')] * len(self.flights)
        parent = [-1] * len(self.flights)
        
        # Initialize starting flights
        for starting_flight in self.cities[start_city]:
            # Check departure time constraint
            if starting_flight.departure_time >= t1 and starting_flight.arrival_time <= t2:
                min_fares[starting_flight.flight_no] = starting_flight.fare
                pq.insert((starting_flight, starting_flight.fare))
        
        while not pq.is_empty():
            current = pq.extract()
            #current cannot be but if the variable inside are none the  dikkat ho sakti hai
            if current is None:
                break
                
            current_flight, current_total_fare = current
            
            # Skip if we've found a better path already
            if current_total_fare > min_fares[current_flight.flight_no]:
                continue
                
            # Check if we've reached destination
            if current_flight.end_city == end_city:
                if current_total_fare < cheapest_total_fare:
                    cheapest_total_fare = current_total_fare
                    
                    # Reconstruct the path
                    path = []
                    currentflight_number = current_flight.flight_no
                    while currentflight_number != -1:
                        path.append(self.flights[currentflight_number])
                        currentflight_number = parent[currentflight_number]
                    path = path[::-1]

                    final_ans = path
                continue
            
            # Explore next flights
            for next_flight in self.flightadj[current_flight.flight_no][1]:
                # Check time constraints for next flight
                if next_flight.arrival_time <= t2:
                    
                    
                    new_fare = current_total_fare + next_flight.fare
                    
                    # Update if we found a cheaper path
                    if new_fare < min_fares[next_flight.flight_no]:
                        min_fares[next_flight.flight_no] = new_fare
                        parent[next_flight.flight_no] = current_flight.flight_no
                        pq.insert((next_flight, new_fare))
                else:
                    continue
        
        return final_ans
    
   
        
        # pass
    def least_flights_cheapest_route(self, start_city, end_city, t1, t2):
        """
        Return List[Flight]: A route from start_city to end_city, which:
        - Departs after t1 (>= t1)
        - Arrives before t2 (<= t2)
        - Has minimum number of flights
        - Among routes with same number of flights, is the cheapest
        """

        if start_city == end_city:
            return []
        final_ans = []
        min_flights = float('inf')
        cheapest_total_fare = float('inf')
        pq = Heap(comparator_function_min)
        
        parent = [-1] * len(self.flights)  # Store flight_no of parent to give ans for back ward
        
        # Tracking minimum flights and fares to reach each flight from current
        min_no_flight = [(float('inf'), float('inf'))] * len(self.flights)  
        # Try each possible starting flight from start_city
        for starting_flight in self.cities[start_city]:
            if starting_flight.departure_time < t1:
                continue
                
            # Initialize for Dijkstra's algorithm priority queue i have implement heap
              # Min-heap based )
            pq.insert((starting_flight, 1, starting_flight.fare))  # (flight, num_flights, total_fare)
            min_no_flight[starting_flight.flight_no] = (1, starting_flight.fare)
            # Tracking visited flights and their parents for path 
        visited = [False] * len(self.flights)
        parent = [-1] * len(self.flights)  # Store flight_no of parent to give ans for back ward
        
        # Tracking minimum flights and fares to reach each flight from current
       
        
        while not pq.is_empty():
            current = pq.extract()
            if current is None:
                break
                
            current_flight, currentflight_totalcount, current_total_fare = current
            
            # if visited[current_flight.flight_no]==True:
            #     continue
            # else :
            #     visited[current_flight.flight_no] = True
            
            # If we've reached end_city,w will  check if this is the best route for final ans
            if current_flight.end_city == end_city and current_flight.arrival_time <= t2:
                if (currentflight_totalcount < min_flights or (currentflight_totalcount == min_flights and current_total_fare < cheapest_total_fare)):
                    min_flights = currentflight_totalcount
                    cheapest_total_fare = current_total_fare
                    
                    # Reconstructing  the path for updated list
                    path = []
                    currentflight_number = current_flight.flight_no
                    while currentflight_number != -1:
                        path.append(self.flights[currentflight_number])
                        currentflight_number = parent[currentflight_number]
                    path = path[::-1]

                    final_ans = path
                continue
            
            # Explore all possible next flights for this flight
            for next_flight in self.flightadj[current_flight.flight_no][1]:
                if next_flight.arrival_time <=t2:
                        
                    new_flight_count = currentflight_totalcount + 1
                    new_fare = current_total_fare + next_flight.fare
                    
                    # Only consider this path if it has fewer flights 
                    curr_min_flights, curr_min_fare = min_no_flight[next_flight.flight_no]
                    if (new_flight_count < curr_min_flights ):
                        min_no_flight[next_flight.flight_no] = (new_flight_count, new_fare)
                        parent[next_flight.flight_no] = current_flight.flight_no
                        pq.insert((next_flight, new_flight_count, new_fare))

                    #   same flights but cheaper which is required for the question
                    elif (new_flight_count == curr_min_flights and new_fare < curr_min_fare):
                        min_no_flight[next_flight.flight_no] = (new_flight_count, new_fare)
                        parent[next_flight.flight_no] = current_flight.flight_no
                        pq.insert((next_flight, new_flight_count, new_fare))
                else:
                    continue
        return final_ans