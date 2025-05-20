import json
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_time_difference(departure_str, arrival_str, halt, days_diff, current_date):
    """Calculates the time difference between departure and arrival, considering halts and day changes."""
    departure_dt = datetime.strptime(f"{current_date} {departure_str}", '%Y-%m-%d %H:%M')
    arrival_dt = datetime.strptime(f"{current_date} {arrival_str}", '%Y-%m-%d %H:%M') + timedelta(days=days_diff)
    time_difference = (arrival_dt - departure_dt).total_seconds() / 3600  # Convert to hours
    return time_difference + halt / 60  # Adding halt time in hours

def calculate_distance_between_stations(station1, station2, distances):
    """Retrieves distance between two stations."""
    return distances.get(station1, {}).get(station2, distances.get(station2, {}).get(station1))

def calculate_average_speed_fragments(schedule, distances):
    """Calculates average speed between consecutive stations, accounting for halts and days."""
    if len(schedule) < 2:
        return None
    
    total_speed, count = 0, 0
    current_date = datetime(2025, 3, 27).strftime('%Y-%m-%d')
    
    for i in range(len(schedule) - 1):
        dep_station, dep_time, halt, dep_days = schedule[i]['station'], schedule[i]['departure'], schedule[i]['halt'], schedule[i]['day']
        arr_station, arr_time, arr_days = schedule[i + 1]['station'], schedule[i + 1]['arrival'], schedule[i + 1]['day']
        
        if dep_time and arr_time:
            days_diff = arr_days - dep_days  # Handling day change
            time_diff_hours = calculate_time_difference(dep_time, arr_time, halt, days_diff, current_date)
            distance = calculate_distance_between_stations(dep_station, arr_station, distances)
            
            if distance and time_diff_hours > 0:
                total_speed += distance / time_diff_hours
                count += 1
    
    return total_speed / count if count else None

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
    
    optimized_schedules = load_json_file(optimized_schedules_file)
    station_distances = load_json_file(station_distances_file)
    
    if not optimized_schedules or not station_distances:
        print("Error: Required JSON files not found or empty.")
        return
    
    avg_fragment_speeds = [
        calculate_average_speed_fragments(schedule, station_distances)
        for schedule in optimized_schedules.values()
        if calculate_average_speed_fragments(schedule, station_distances) is not None
    ]
    
    if avg_fragment_speeds:
        plt.figure(figsize=(10, 6))
        sns.kdeplot(avg_fragment_speeds, color='blue', fill=True, alpha=0.6)
        plt.title('Distribution of Average Speeds (Considering Halts & Days)')
        plt.xlabel('Average Speed (km/h)')
        plt.ylabel('Density')
        plt.grid(axis='y', alpha=0.75)

        # Define the output path for the plot
        output_image_path = r'C:\Users\harad\Desktop\TrainData\average_fragment_speed_distribution.png'
        
        # Delete the file if it already exists to ensure it can be overwritten
        if os.path.exists(output_image_path):
            os.remove(output_image_path)
        
        # Save the plot (this will now overwrite the existing file if present)
        plt.savefig(output_image_path)
        print(f"Plot saved to {output_image_path}")
    else:
        print("No valid average speed data found.")

if __name__ == "__main__":
    main()
