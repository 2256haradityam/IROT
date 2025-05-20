import json
import os
def load_optimized_schedules(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def extract_schedule_data(schedules):
    schedule_list = []
    for train_number, details in schedules.items():
        for entry in details:
            schedule_list.append({
                'TRAIN_NUMBER': train_number,
                'TRAIN_NAME': entry.get('train_name', ''),
                'STATION_NAME': entry.get('station', ''),
                'ARRIVAL_TIME': entry.get('arrival', ''),
                'DEPARTURE_TIME': entry.get('departure', ''),
                'HALT_TIME': entry.get('halt', 0),
                'DAY': entry.get('day', 0)
            })
    return schedule_list