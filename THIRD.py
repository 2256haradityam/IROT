import json
from datetime import datetime, timedelta
from collections import defaultdict, deque
import pytz
import heapq
import os
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
import sys
import time
import math
# --- CORE.py logic begins here ---

class TrainScheduler:
    def __init__(self, datasets):
        self.load_data(datasets)
        self.initialize_state()
        self.central_clock = datetime(2024, 1, 1, 0, 0, tzinfo=pytz.timezone('Asia/Kolkata'))
        self.max_speed = int(os.getenv('MAX_SPEED', 130))
        self.high_priority_speed = int(os.getenv('HIGH_PRIORITY_SPEED', 130))
        self.medium_priority_speed = int(os.getenv('MEDIUM_PRIORITY_SPEED', 100))
        self.low_priority_speed = int(os.getenv('LOW_PRIORITY_SPEED', 80))
    def load_data(self, datasets):
        with open(datasets['train_stations']) as f:
            self.train_routes = json.load(f)

        with open(datasets['station_distances']) as f:
            self.distances = json.load(f)

        with open(datasets['train_station_info']) as f:
            self.station_info = json.load(f)

        with open(datasets['train_speeds']) as f:
            self.original_speeds = json.load(f)
        self.original_speeds_dict = {item.split(',:')[0]: float(item.split(',: ')[1].split(' ')[0]) for item in self.original_speeds}

    def initialize_state(self):
        self.schedules = {}
        self.platform_occupancy = defaultdict(list)
        self.departure_history = defaultdict(deque)
        self.time_windows = {
            'high_priority': [(9, 15), (21, 3)],
            'medium_priority': [(9, 15), (21, 3)],
            'low_priority': [(3, 9), (15, 21)]
        }
        self.train_queue = []
        heapq.heapify(self.train_queue)
        for train_id in self.train_routes:
            heapq.heappush(self.train_queue, (self.get_train_priority(train_id), train_id))

    def get_train_priority(self, train_number):
        if train_number.startswith(('12', '22')):
            return 3  # High priority
        elif train_number.startswith(('1', '2')):
            return 2  # Medium priority
        return 1  # Low priority

    def get_train_category(self, train_number):
        if train_number.startswith(('12', '22')):
            return 'high_priority'
        elif train_number.startswith(('1', '2')):
            return 'medium_priority'
        return 'low_priority'
    
    
    def calculate_target_speed(self, category):
     
     return {
        'high_priority': self.high_priority_speed,
        'medium_priority': self.medium_priority_speed,
        'low_priority': self.low_priority_speed
     }[category]

    def adjust_to_window(self, category, proposed_time):
        windows = self.time_windows[category]
        for start, end in windows:
            if (start <= proposed_time.hour < end) or (end < start and (proposed_time.hour >= start or proposed_time.hour < end)):
                return proposed_time
        
        next_window = min(
            [w for w in windows], 
            key=lambda w: (w[0] - proposed_time.hour) % 24
        )
        days = 1 if next_window[0] < proposed_time.hour else 0
        return proposed_time.replace(
            hour=next_window[0], minute=0, second=0
        ) + timedelta(days=days)

    def calculate_travel_time(self, distance, category, train_number):
     target_speed = self.calculate_target_speed(category)
     original_speed = self.original_speeds_dict.get(train_number, target_speed)
     speed = min(self.max_speed, max(target_speed, original_speed))
     return math.ceil(((distance / speed)*60) + ((speed*2)/60))  # minutes

    def handle_platform_occupancy(self, station, arrival_time):
        station_data = self.station_info.get(station, {})
        max_platforms = station_data.get('max_platform', 2)
        
        available_platforms = [t for t in self.platform_occupancy[station] if t <= arrival_time]
        
        while self.platform_occupancy[station] and self.platform_occupancy[station][0] <= arrival_time:
            heapq.heappop(self.platform_occupancy[station])

        if len(self.platform_occupancy[station]) >= max_platforms:
            return True, self.platform_occupancy[station][0]

        return False, None
    
    def delay_at_previous_station(self, train_id, current_station_index, schedule, delay_minutes):
        if current_station_index > 0:
            prev_station_schedule = schedule[current_station_index - 1]
            
            if prev_station_schedule:
                prev_departure_str = prev_station_schedule.get('departure')
                if prev_departure_str and prev_departure_str != "DESTINATION":
                    prev_departure_time = datetime.strptime(prev_departure_str, '%H:%M').replace(tzinfo=self.central_clock.tzinfo,year=self.central_clock.year, month = self.central_clock.month, day = self.central_clock.day)
                    new_departure_time = prev_departure_time + timedelta(minutes=delay_minutes)
                    schedule[current_station_index - 1]['departure'] = new_departure_time.strftime('%H:%M')
                    
                    for station_schedule in schedule[current_station_index:]:
                        if station_schedule.get('arrival') and station_schedule.get('arrival')!= "DESTINATION":
                            arrival_time = datetime.strptime(station_schedule.get('arrival'), '%H:%M').replace(tzinfo=self.central_clock.tzinfo, year=self.central_clock.year, month=self.central_clock.month, day = self.central_clock.day)
                            new_arrival_time = arrival_time + timedelta(minutes=delay_minutes)
                            station_schedule['arrival'] = new_arrival_time.strftime('%H:%M')
                        if station_schedule.get('departure') and station_schedule.get('departure') != "DESTINATION":
                            departure_time = datetime.strptime(station_schedule.get('departure'), '%H:%M').replace(tzinfo=self.central_clock.tzinfo, year=self.central_clock.year, month = self.central_clock.month, day = self.central_clock.day)
                            new_departure_time = departure_time + timedelta(minutes=delay_minutes)
                            station_schedule['departure'] = new_departure_time.strftime('%H:%M')

    def schedule_train(self, train_id):
        route = self.train_routes[train_id]
        category = self.get_train_category(train_id)
        schedule = []
        current_time = self.central_clock

        for i, station in enumerate(route):
            if i == 0:
                # Adjust arrival time based on the priority window
                current_time = self.adjust_to_window(category, current_time)

                if self.departure_history[station]:
                    last_departure = self.departure_history[station][-1]
                    min_departure = last_departure + timedelta(minutes=5)
                    current_time = max(current_time, min_departure)

                self.departure_history[station].append(current_time)

                schedule.append({
                    'station': station,
                    'arrival': None,
                    'departure': current_time.strftime('%H:%M'),
                    'halt': 0,  # Initial halt is zero, it will be updated later
                    'day': current_time.day
                })
                continue

            prev_station = route[i - 1]
            distance = self.distances.get(prev_station, {}).get(station, 0)
            travel_minutes = self.calculate_travel_time(distance, category, train_id)
            arrival_time = current_time + timedelta(minutes=travel_minutes)

            # Handle platform occupancy
            is_occupied, earliest_available = self.handle_platform_occupancy(station, arrival_time)

            if is_occupied:
                self.delay_at_previous_station(train_id, i, schedule, 5)
                arrival_time = datetime.strptime(schedule[i - 1]['departure'], '%H:%M').replace(
                    tzinfo=self.central_clock.tzinfo, year=self.central_clock.year, month=self.central_clock.month,
                    day=self.central_clock.day) + timedelta(minutes=travel_minutes)

                # Check again for platform occupancy after delay
                is_occupied, earliest_available = self.handle_platform_occupancy(station, arrival_time)
                if is_occupied:
                    self.delay_at_previous_station(train_id, i - 1, schedule, 5)
                    arrival_time = datetime.strptime(schedule[i - 1]['departure'], '%H:%M').replace(
                        tzinfo=self.central_clock.tzinfo, year=self.central_clock.year, month=self.central_clock.month,
                        day=self.central_clock.day) + timedelta(minutes=travel_minutes)

            # Ensure the halt time is at least the minimum halt time from the input file
            original_halt = None
            for train_data in self.station_info.get(station, {}).get('trains', []):
                train_number = train_data['train_number'].strip()
                train_id_trimmed = train_id.split()[0].strip()
                if train_number == train_id_trimmed:
                    original_halt = train_data['halt']
                    break
            
            if original_halt is None:
                original_halt = 1

            departure_time = arrival_time + timedelta(minutes=original_halt)

            # Ensure platform occupancy after assigning departure time
            heapq.heappush(self.platform_occupancy[station], departure_time)

            schedule.append({
                'station': station,
                'arrival': arrival_time.strftime('%H:%M'),
                'departure': departure_time.strftime('%H:%M'),
                'halt': original_halt,
                'day': arrival_time.day
            })

            current_time = departure_time

        self.schedules[train_id] = schedule

    def run_simulation(self):
        while self.train_queue:
            _, train_id = heapq.heappop(self.train_queue)
            self.schedule_train(train_id)
        return self.schedules

# Input files for CORE
datasets = {
    'train_stations': 'train_stations.json',
    'station_distances': 'station_distances.json',
    'train_station_info': 'train_station_info_with_halt_excluded_lowercase.json',
    'train_speeds': 'train_speeds.json'
}
st = time.time()
# Running CORE logic
scheduler = TrainScheduler(datasets)
optimized_schedules = scheduler.run_simulation()

# Saving the optimized schedules to a JSON file
output_path = 'optimized_schedules.json'
with open(output_path, 'w') as f:
    json.dump(optimized_schedules, f, indent=2)
en = time.time()
print(f"CORE logic executed in {en - st:.2f} seconds.")
print(f"Optimization complete. Schedules saved to {os.path.abspath(output_path)}")

# --- GRAPH.py logic begins here ---

def calculate_time_difference(departure_str, arrival_str, halt, days_diff, current_date):
    """Calculates the time difference between departure and arrival, considering halts and day changes."""
    departure_dt = datetime.strptime(f"{current_date} {departure_str}", '%Y-%m-%d %H:%M')
    arrival_dt = datetime.strptime(f"{current_date} {arrival_str}", '%Y-%m-%d %H:%M') + timedelta(days=days_diff)
    time_difference = (arrival_dt - departure_dt).total_seconds() / 3600  # Convert to hours
    return time_difference + halt / 60  # Adding halt time in hours

def update_halt_times(optimized_schedules):
    """Updates halt times for each station in the schedules."""
    for train, schedule in optimized_schedules.items():
        for i in range(1, len(schedule)):
            dep_time_str = schedule[i]['departure']
            arr_time_str = schedule[i]['arrival']
            
            if dep_time_str and arr_time_str:
                # Calculate the halt time as difference between departure and arrival
                dep_time = datetime.strptime(dep_time_str, '%H:%M')
                arr_time = datetime.strptime(arr_time_str, '%H:%M')
                if dep_time > arr_time:
                 halt_time = int((dep_time - arr_time).total_seconds() / 60 if arr_time else 0) # Halt time in minutes
                else:
                    arr_time -= timedelta(days=1)  # Adjust for next day arrival
                    halt_time = int((dep_time - arr_time).total_seconds() / 60 if arr_time else 0 )# Halt time in minutes
                schedule[i]['halt'] = halt_time  # Update halt time for the station

    return optimized_schedules


def calculate_distance_between_stations(station1, station2, distances):
    """Retrieves distance between two stations."""
    return distances.get(station1, {}).get(station2) or distances.get(station2, {}).get(station1)

def calculate_average_speed_fragments(schedule, distances):
    """Returns average speed between each pair of stations."""
    if len(schedule) < 2:
        return None

    total_speed = 0
    count = 0
    current_date = '2025-03-27'

    for i in range(len(schedule) - 1):
        s1, s2 = schedule[i], schedule[i+1]
        dep_time = s1.get('departure')
        arr_time = s2.get('arrival')
        halt = s1.get('halt', 0)
        day_diff = s2.get('day', 1) - s1.get('day', 1)

        if dep_time and arr_time:
            try:
                hours = calculate_time_difference(dep_time, arr_time, halt, day_diff, current_date)
                distance = calculate_distance_between_stations(s1['station'], s2['station'], distances)
                if distance and hours > 0:
                    total_speed += distance / hours
                    count += 1
            except:
                continue
    return total_speed / count if count > 0 else None

def load_json_file(filepath):
    """Loads JSON data from a file safely."""
    if os.path.exists(filepath):
        with open(filepath, 'r') as file:
            return json.load(file)
    return None

def main():
    # Update file paths to Windows-compatible paths
    optimized_schedules_file = 'optimized_schedules.json'
    station_distances_file = 'station_distances.json'
    
    # Load the JSON files
    optimized_schedules = load_json_file(optimized_schedules_file)
    station_distances = load_json_file(station_distances_file)
    
    if not optimized_schedules or not station_distances:
        print("Error: Required JSON files not found or empty.")
        return
    optimized_schedules = update_halt_times(optimized_schedules)

    # Reopen the file in write mode and save the updated schedules
    with open(optimized_schedules_file, 'w') as f:
        json.dump(optimized_schedules, f, indent=2)
    print(f"Updated schedules saved to {os.path.abspath(optimized_schedules_file)}")
    # Calculate average speed for each fragment
    avg_fragment_speeds = []
    for schedule in optimized_schedules.values():
        speed = calculate_average_speed_fragments(schedule, station_distances)
        if speed:
            avg_fragment_speeds.append(speed)

    # Plot results
    if avg_fragment_speeds:
        plt.figure(figsize=(10, 6))
        plt.hist(avg_fragment_speeds, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        plt.title('Distribution of Average Speeds of All Trains\n(Considering Halts & Day Changes)', fontsize=14)
        plt.xlabel('Average Speed (km/h)')
        plt.ylabel('Number of Trains')
        plt.grid(True, axis='y', linestyle='--', alpha=0.6)

        output_image_path = 'average_fragment_speed_distribution.png'
        plt.savefig(output_image_path)
        plt.close()
        print(f"Plot saved to {output_image_path}")
    else:
        print("No valid average speed data found.")

if __name__ == "__main__":
    main()