from flight import Flight
from planner import Planner
def find_optimized_paths(flights, start_city, end_city, t1, t2):
    """
    Finds and returns optimized paths from start_city to end_city based on:
    - Least flights and earliest arrival
    - Least flights and cheapest fare
    - Overall cheapest route
    
    Args:
        flights (List[Flight]): List of available flights.
        start_city (int): Starting city.
        end_city (int): Destination city.
        t1 (int): Earliest departure time.
        t2 (int): Latest arrival time.
    
    Returns:
        Dict: A dictionary containing paths for the specified optimization criteria.
    """
    all_paths = []  # To store all valid paths

    def dfs(current_city, current_path, total_time, total_fare, flights_taken, prev_flight):
        # If we reached the end city, store the current path details
        if current_city == end_city:
            all_paths.append({
                "path": current_path[:],
                "total_time": total_time,
                "total_fare": total_fare,
                "num_flights": flights_taken
            })
            return

        # Explore all flights starting from the current city
        for flight in flights:
            # Consider only flights originating from the current city
            if flight.start_city != current_city:
                continue

            # Check departure constraint for the first flight
            if prev_flight is None and flight.departure_time < t1:
                continue
            
            # Ensure layover time of at least 20 minutes between flights
            if prev_flight is not None and flight.departure_time < prev_flight.arrival_time + 20:
                continue
            
            # Check if the flight's arrival is within the allowed time window
            if flight.arrival_time > t2:
                continue
            
            # Calculate new path details
            new_time = total_time + (flight.arrival_time - flight.departure_time)
            new_fare = total_fare + flight.fare
            new_flight_count_taken = flights_taken + 1

            # Recur to explore further paths
            current_path.append(flight)
            dfs(flight.end_city, current_path, new_time, new_fare, new_flight_count_taken, flight)
            current_path.pop()  # Backtrack

    # Start DFS from the start city
    dfs(start_city, [], 0, 0, 0, None)

    # Variables to store the optimized paths
    least_flights_least_time = None
    least_flights_cheapest_fare = None
    overall_cheapest = None

    min_flights = float('inf')
    min_cost = float('inf')
    min_arrival_time = float('inf')  # Track the earliest arrival time

    for path_info in all_paths:
        last_flight = path_info["path"][-1] if path_info["path"] else None
        
        if path_info["num_flights"] < min_flights:
            min_flights = path_info["num_flights"]
            least_flights_least_time = path_info
            least_flights_cheapest_fare = path_info
            min_arrival_time = last_flight.arrival_time if last_flight else float('inf')
        elif path_info["num_flights"] == min_flights:
            # Update least flights and cheapest fare path if fare is lower
            if path_info["total_fare"] < least_flights_cheapest_fare["total_fare"]:
                least_flights_cheapest_fare = path_info
            # Update least flights and least time if arrival time is earlier
            if last_flight and last_flight.arrival_time < min_arrival_time:
                least_flights_least_time = path_info
                min_arrival_time = last_flight.arrival_time

        # Overall cheapest path (regardless of flights taken)
        if path_info["total_fare"] < min_cost:
            min_cost = path_info["total_fare"]
            overall_cheapest = path_info

    # Create result dictionary to hold the paths
    result = {
        "least_flights_least_time": least_flights_least_time,
        "least_flights_cheapest_fare": least_flights_cheapest_fare,
        "overall_cheapest": overall_cheapest
    }

    # Write paths to a file
    with open('optimized_paths.txt', 'w') as file:
        for key, path_info in result.items():
            if path_info:
                file.write(f"{key.replace('_', ' ').capitalize()}:\n")
                for flight in path_info["path"]:
                    file.write(f"  Flight No: {flight.flight_no}, From: {flight.start_city} To: {flight.end_city}, "
                               f"Departure: {flight.departure_time}, Arrival: {flight.arrival_time}, Fare: {flight.fare}\n")
                file.write(f"  Total Time: {path_info['total_time']} minutes\n")
                file.write(f"  Total Fare: {path_info['total_fare']}\n")
                file.write(f"  Number of Flights: {path_info['num_flights']}\n\n")
            else:
                file.write(f"{key.replace('_', ' ').capitalize()}: No valid path found.\n\n")

    return result


def main():
    flights = [ Flight(0,0,10,1,30,10),
                Flight(1,0,10,2,20,50),  
                Flight(2,1,50,2,70,10),  
                Flight(3,1,40,3,60,40),  
                Flight(4,1,90,4,140,40),  
                Flight(5,2,40,3,50,30),  
                Flight(6,2,90,3,110,10),  
                Flight(7,2,20,4,40,20),  
                Flight(8,3,130,4,150,10),  
                Flight(9,3,70,5,80,20),  
                Flight(10,4,180,5,200,10),  
               ]
    flights3 = [
                Flight(0,0,300,1,330,100),
                Flight(1,0,360,1,390,10),
                Flight(2,1,350,2,400,10),
                Flight(3,1,410,2,420,1000),
    ]
    # flights2 = [    Flight(0, 0, 5, 1, 25, 15),
    # Flight(1, 0, 10, 2, 30, 40),
    # Flight(2, 0, 15, 3, 40, 20),
    # Flight(3, 1, 35, 2, 55, 25),
    # Flight(4, 1, 50, 3, 70, 30),
    # Flight(5, 1, 55, 4, 100, 50),
    # Flight(6, 2, 60, 3, 90, 15),
    # Flight(7, 2, 70, 4, 95, 35),
    # Flight(8, 3, 80, 4, 100, 25),
    # Flight(9, 3, 90, 5, 120, 45),
    # Flight(10, 4, 110, 5, 130, 20),
    # Flight(11, 4, 140, 6, 180, 60),
    # Flight(12, 5, 150, 6, 200, 30),
    # Flight(13, 6, 160, 7, 210, 70),
    # Flight(14, 6, 180, 8, 230, 50),
    # Flight(15, 7, 200, 8, 240, 40),
    # Flight(16, 7, 220, 9, 260, 80),
    # Flight(17, 8, 250, 9, 290, 35),
    # Flight(18, 9, 280, 10, 310, 60),
    # Flight(19, 9, 300, 10, 340, 20)  
    #            ]

    flight_planner1 = Planner(flights3)
    flight_planner2 = Planner(flights)

    # The three tasks
    route1 = flight_planner1.least_flights_earliest_route(0,2,0,1000)
    route2 = flight_planner1.cheapest_route(0,2,0,1000)
    route3 = flight_planner1.least_flights_cheapest_route(0,2,0,1000)
    paths = find_optimized_paths(flights3,0,2,0,1000)

    
    # model output
    expected_route1 = paths['least_flights_least_time']['path']             
    expected_route2 = paths["overall_cheapest"]['path']   
    expected_route3 = paths['least_flights_cheapest_fare']['path']       
    
    # Note that for this given example there is a unique solution, but it may
    # not be true in general
    # print(route1)
    # print(route2)
    # print(route3)
    if route1 == expected_route1:
        print("Task 1 PASSED")
        
    if route2 == expected_route2:
        print("Task 2 PASSED")
        
    if route3 == expected_route3:
        print("Task 3 PASSED")

    route1 = flight_planner2.least_flights_earliest_route(0,5,0,1000)
    route2 = flight_planner2.cheapest_route(0,5,0,1000)
    route3 = flight_planner2.least_flights_cheapest_route(0,5,0,1000)
    paths = find_optimized_paths(flights,0,5,0,1000)
    expected_route1 = paths['least_flights_least_time']['path']             
    expected_route2 = paths["overall_cheapest"]['path']   
    expected_route3 = paths['least_flights_cheapest_fare']['path']       
    
    # Note that for this given example there is a unique solution, but it may
    # not be true in general
    # print(route1)
    # print(route2)
    # print(route3)
    if route1 == expected_route1:
        print("Task 4 PASSED")
        
    if route2 == expected_route2:
        print("Task 5 PASSED")
        
    if route3 == expected_route3:
        print("Task 6 PASSED")

if __name__ == "__main__":
    main()