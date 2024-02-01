import librosa
import numpy as np
import json
from pathlib import Path
import pychorus
import math
import quiet_before_drop
from pathlib import Path
from typing import Union, List
from dataclasses import asdict
import numpy as np
import DataService

def get_rms(song_data=None, category=None, path=None)->tuple[list[float], float]:
    # Load the segment data from the JSON file
    if path:
        data_path = path
    elif category:
        data_path = song_data['demixed']
    elif song_data:
        data_path = song_data['file']
    else:
        raise Exception("Invalid arguments")
    # Calculate the average intensity
    if category:
        list = []
        for instru in category:
            path = f"{data_path}{instru}.wav"
            data_y, sr = librosa.load(path)
            list.append(librosa.feature.rms(y=data_y)[0])
        rms = [sum(x) for x in zip(*list)]
        average = sum(rms) / len(rms)
        return rms, average
    data_y, sr = librosa.load(data_path)
    rms = librosa.feature.rms(y=data_y)[0]
    average = sum(rms) / len(rms)
    return rms, average

def section_by_rms(song_data=None, category=None, path=None, name=None, pauses=None):
    # Load the RMS and average
    if path:
        rms, average = get_rms(path=Path(path))
    elif category:
        print(category)
        rms, average = get_rms(song_data, category=category)
    else:
        rms, average = get_rms(song_data)
    rms_average = get_modified_rms(song_data, name=name, rms=rms, pauses=pauses)
    average = sum(rms_average) / len(rms_average)
    song_rms = get_rms(song_data)[0]
    update_struct(song_data, name=name, rms=rms)

    # Define the window size and the minimum number of frames
    window_size = 50
    min_frames = 40
    high_scale = 0.65

    # Define thresholds for high, medium and low volume
    high_threshold = average * 0.5
    low_threshold = average * 0.25

    # Initialize a list to hold the sections
    sections = []

    # Initialize a variable to hold the current section
    current_section = None
    current_category = None

    # Initialize the window start index
    i = 0
    # While there is enough data left for a window
    while i <= len(rms) - window_size:
        # Determine the volume category of the window
        quiet_count = sum(1 for j in range(i, i + window_size) if rms[j] <= low_threshold)
        loud_count = sum(1 for j in range(i, i + window_size) if rms[j] >= high_threshold)
        avg_volume = sum(song_rms[j] for j in range(i, i + window_size)) / window_size

        if loud_count >= high_scale * min_frames:
            volume_category = 'loud'
        elif loud_count >= min_frames:
            volume_category = 'loud'
        elif quiet_count >= min_frames:
            volume_category = 'quiet'
        else:
            volume_category = 'medium'

        # If this is the first window or the volume category has changed
        if current_section is None or volume_category != current_category:
            # If there is a current section, add it to the list of sections
            if current_section is not None:
                sections.append(current_section)

            # Start a new section
            current_section = {'start': i, 'end': i + window_size - 1, 'category': volume_category, 'avg_volume': avg_volume}
            current_category = volume_category

        # Otherwise, continue the current section
        else:
            current_section['end'] = i + window_size - 1
            current_section['avg_volume'] = (current_section['avg_volume'] * (window_size - 1) + song_rms[i + window_size - 1]) / window_size

        # Move the window start to the end of the current section
        i = current_section['end'] + 1

    # Add the last section to the list of sections
    if current_section is not None:
        sections.append(current_section)

    return sections

def update_struct(song_data, name=None, category=None, path=None, rms=False, params=[]):
    params = params
    struct_data = DataService.get_struct_data(name)
    if rms != False:
        if type(rms) != list:
            rms, average = get_rms(song_data, category=category)
        segments = struct_data['segments']
        for segment in segments:
            segment_start = segment['start']
            segment_end = segment['end']
            average_rms = sum(rms[int(segment_start*43):int(segment_end*43)]) / len(rms[int(segment_start*43):int(segment_end*43)])

            segment['avg_volume'] = average_rms

        params.append({'segments': segments})

    DataService.update_struct_data(name=name, params=params)

#INSERT SPAGHETTI CODE HERE (I'm sorry)
def segment(name, song_data, sectionby):
    # Use section_by_rms to segment the song into sections for vocals and for drums/other
    pauses = quiet_before_drop.get_pauses(name, song_data)
    if not sectionby:
        rms_ = section_by_rms(song_data, name=name, pauses=pauses)
    else:
        rms_ = section_by_rms(song_data, category=sectionby, name=name, pauses=pauses)
    rms_short = merge_short_sections(rms_)
    rms_sections = merge_same_category_sections(rms_short)
    update_struct(song_data, name=name, params=[{'sections1': rms_, 'sections2': rms_short, 'sections3': rms_sections}])

    # Load the structure data from the JSON file
    struct_data = DataService.get_struct_data(name)

    # Find the start times of the choruses
    segments = struct_data['segments']
    threshold1 = 405
    threshold2 = -405
    threshold3 = 405
    # Identify the sections where choruses begin
    chorus_sections = []
    i = 0
    for segment in segments:
        if segment['label'] == 'start' or segment['label'] == 'verse' or segment['label'] == 'bridge' or segment['label'] == 'outro':
            i+=1
            continue
        # if i > 0:
        #     if segments[i-1]["start"] in chorus_sections:
        #         i+=1
        #         continue
        segment_start = segment['start']
        segment_end = segment['end']
        segment_str = str(segment_start)
        for section in rms_sections:
            go = False
            if section['end'] - section['start'] <= 395:
                continue
            start_difference = 43 * segment_start - section['start']
            end_difference = abs(43 * segment_end - section['end'])
            if start_difference <= 0 and start_difference >= threshold2:
                go = True
                why = 1
            elif start_difference >= 0 and start_difference <= threshold1:
                go = True
                why = 2
            elif end_difference <= threshold3:
                go = True
                why = 3
            elif end_difference < 0 and end_difference >= threshold2:
                go = True
                why = 4            
            if go:
                print(f"Segment start: {segment_start} at {section} being checked")
                if section['category'] == 'loud' and segment_start not in chorus_sections:
                    # Check if the start of the previous section is not in chorus_sectionsw
                    if i < len(segments) - 1 and segments[i+1]['label'] != 'inst':
                            if segments[i-1]['start'] not in chorus_sections:
                                why = 5
                                chorus_sections.append(segment_start)
                                print(f"Segment start: {segment_start} added by loud at {section} Start difference: {start_difference} End difference: {end_difference} why: {why}")
                                break
                        # elif abs(start_difference) <= 100 and (i > 0 and segments[i-1]['start'] in chorus_sections) and (section_ids.get(segment_str, (0, 0))[1] - section_ids.get(segment_str, (0, 0))[0] >= 395):
                        #     chorus_sections.remove(segments[i-1]['start'])
                        #     chorus_sections.append(segment_start)
                        #     section_ids[segment_str] = (section['start'], section['end'])
                        #     print(f"Segment start: {segment_start} added by loud at {section} Start difference: {start_difference} End difference: {end_difference} why: {why}")
                        #     break
                            elif i == 0:
                                chorus_sections.append(segment_start)
                                why = 6
                                print(f"Segment start: {segment_start} added by loud at {section} Start difference: {start_difference} End difference: {end_difference} why: {why}")
                                break
                            if segment["label"] == "inst":
                                chorus_sections.append(segment_start)
                                why =7
                                print(f"Segment start: {segment_start} added by loud at {section} Start difference: {start_difference} End difference: {end_difference} why: {why}")
                                break
                    elif abs(start_difference) >= 100:
                        chorus_sections.append(segment_start)
                        why = 8
                        print(f"Segment start: {segment_start} added by loud at {section} Start difference: {start_difference} End difference: {end_difference} why: {why}")
                        break
            elif section['start'] <= 43 * segment_start <= section['end'] and section['category'] == 'loud':
                for pause in pauses:
                    if section["end"] - pause[0] <= 160 or (segment["label"] == "chorus" and segments[i + 1]["label"] != "inst"):
                        break
                    if pause[1] - pause[0] <= 160:
                        if (abs(pause[0] - section["start"]) <= 100 or abs(pause[1] - section["start"]) <= 100) and ((abs(pause[0] - 43*segment_start) <= 100 or abs(pause[1] - 43*segment_start) <= 100) and segment["label"] == "chorus" or segment["label"] == "inst"):
                            if segment_start not in chorus_sections:
                                chorus_sections.append(segment_start)
                                print(f"Segment start: {segment_start} added by pause {pause} at {section}")
                break
        i += 1
    return chorus_sections

def merge_short_sections(sections):
    merged_sections = []
    temp_section = sections[0]

    for i in range(1, len(sections)):
        current_section = sections[i]

        # If the current section is short, extend the temp section
        if current_section['end'] - temp_section['start'] <= 400:
            temp_section['end'] = current_section['end']
        else:
            # If the current section is not short, add the temp section to the merged sections and start a new temp section
            merged_sections.append(temp_section)
            temp_section = current_section

    # If there is a temp section at the end, add it to the merged sections
    merged_sections.append(temp_section)

    return merged_sections

def merge_same_category_sections(sections):
    merged_sections = []
    temp_section = sections[0]

    for i in range(1, len(sections)):
        current_section = sections[i]

        # If the current section has the same category as the temp section, extend the temp section
        if current_section['category'] == temp_section['category']:
            temp_section['end'] = current_section['end']
        else:
            # If the current section has a different category, add the temp section to the merged sections and start a new temp section
            merged_sections.append(temp_section)
            temp_section = current_section

    # If there is a temp section at the end, add it to the merged sections
    merged_sections.append(temp_section)

    return merged_sections

def get_modified_rms(song_data, name, rms=False, category=None, pauses=False):
    if type(rms) != list:
        rms, average = get_rms(song_data, category=category)
    else:
        rms = rms

    if type(pauses) != list:
        pauses = quiet_before_drop.get_pauses(name, song_data)
    else:
        pauses = pauses

    # Initialize the modified RMS list
    modified_rms = rms

    # Set the RMS values between the pauses to 0
    print(f"RMS {len(modified_rms)} Pauses {pauses}")
    for pause in pauses:
        start, end = pause[0], pause[1]
        for i in range(start, end):
            modified_rms[i] = 0

    return modified_rms