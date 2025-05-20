import json
from datetime import datetime, timedelta
from collections import defaultdict, deque
import pytz
import heapq
import os

class TrainScheduler:
    def __init__(self, datasets):
        self.load_data(datasets)
        self.initialize_state()
        self.central_clock = datetime(2024, 1, 1, 0, 0, tzinfo=pytz.timezone('Asia/Kolkata'))

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
            'high_priority': 130,
            'medium_priority': 100,
            'low_priority': 80
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
        max_speed_permitted = 130
        speed = min(max_speed_permitted, max(target_speed, original_speed))
        return distance / speed * 60  # minutes

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
                            arrival_time = datetime.strptime(station_schedule.get('arrival'), '%H:%M').replace(tzinfo=self.central_clock.tzinfo, year=self.central_clock.year, month = self.central_clock.month, day = self.central_clock.day)
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
            # We iterate over the trains at the station to find the current train's halt time
            for train_data in self.station_info.get(station, {}).get('trains', []):
                train_number = train_data['train_number'].strip()
                train_id_trimmed = train_id.split()[0].strip()
                if train_number == train_id_trimmed:
                    original_halt = train_data['halt']
                    break
            
            if original_halt is None:
                original_halt = 1

            # Ensure the halt time is applied correctly, i.e., it's the same as the input halt time
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

# Input files from the local PC
datasets = {
    'train_stations': 'train_stations.json',  # Replace with actual path
    'station_distances': 'station_distances.json',  # Replace with actual path
    'train_station_info': 'train_station_info_with_halt_excluded_lowercase.json',  # Replace with actual path
    'train_speeds': 'train_speeds.json'  # Replace with actual path
}

scheduler = TrainScheduler(datasets)
optimized_schedules = scheduler.run_simulation()

# Save the optimized schedules to local PC and overwrite the file if it exists
output_path = 'optimized_schedules.json'  # Replace with desired output path
with open(output_path, 'w') as f:
    json.dump(optimized_schedules, f, indent=2)

print(f"Optimization complete. Schedules saved to {os.path.abspath(output_path)}")
