"""
Copy and label image files
Set interval minutes, then copy selected files to "figures" directory.
"""

import argparse
import os
from shutil import copyfile

__author__ = "Gyubin Son"
__copyright__ = "Copyright 2018, Daewoo Shipbuilding & Marine Engineering Co., LTd."
__credits__ = ["Minsik Park", "Jaeyun Jeong", "Heejeong Choi", "Gyubin Son", "Jeongho Lee"]
__version__ = "1.0"
__maintainer__ = "Gyubin Son"
__email__ = "gyubin_son@korea.ac.kr"
__status__ = "Develop"


parser = argparse.ArgumentParser(description="Label images and make csv data.")
parser.add_argument("interval", metavar="I", type=int,
                    help="Interval minutes for labeling interval",
                    default=10, nargs='?')
parser.add_argument("data_type", metavar="K", type=int,
                    help="Data type(1st or 2nd or 3rd)",
                    default=1, nargs='?')
parser.add_argument("src_dir", metavar="S", type=str,
                    help="Source directory of images",
                    default="/media/d1", nargs='?')
parser.add_argument("dest_dir", metavar="D", type=str,
                    help="Destination directory of processed images",
                    default="./figures", nargs='?')
parser.add_argument("meta_data_dir", metavar="M", type=str,
                    help="Meta data directory that contains wave, weather, filter data",
                    default="./meta_data", nargs='?')

# Arguments
args = parser.parse_args()
INTERVAL = args.interval
DATA_TYPE = args.data_type
SRC_DIR = args.src_dir
DEST_DIR = args.dest_dir
META_DATA_DIR = args.meta_data_dir

# Meta data
WAVE_DATA_PATH = os.path.join(os.path.abspath(META_DATA_DIR), 'wave.csv')
WEATHER_DATA_PATH = os.path.join(os.path.abspath(META_DATA_DIR), 'weather.csv')
FILTER_DATA_PATH = os.path.join(os.path.abspath(META_DATA_DIR), 'filter_data.csv')

# Number of files
NUM_1ST = 60253
NUM_2ND = 232708
NUM_3RD = 67256
if DATA_TYPE == 1:
    NUM_DATA = NUM_1ST
elif DATA_TYPE == 2:
    NUM_DATA = NUM_2ND
else:
    NUM_DATA = NUM_3RD


def make_date_string(month, day, hour, minutes):
    return '{:02d}/{:02d} {:02d}:{:02d}'.format(month, day, hour, minutes)


def does_data_pass_filter(data, filter_value, data_type):
    data_str = make_date_string(data['month'], data['day'], data['hour'], data['minutes'])
    fv_start = make_date_string(filter_value['month'], filter_value['day'],
                                filter_value['start_hour'], filter_value['start_min'])
    fv_end = make_date_string(filter_value['month'], filter_value['day'],
                              filter_value['end_hour'], filter_value['end_min'])
    
    if filter_value['type'] is not data_type or data_str > fv_end:
        return 'NEXT_FILTER'
    elif data_str < fv_start:
        return 'NEXT_DATA'
    elif data_str >= fv_start and data_str <= fv_end:
        return 'GOOD'
    
    
def get_times_separately(is_filter, time_str):
    if is_filter:
        data = list(map(int, time_str.split(',')))
        result = {
            'type': data[0],
            'year': data[1],
            'month': data[2],
            'day': data[3],
            'start_hour': data[4],
            'start_min': data[5],
            'end_hour': data[6],
            'end_min': data[7]
        }
        return result
    else:
        dates, times = time_str.split(' ')
        month, day, year = map(int, dates.split('/'))
        hour, minutes = map(int, times.split(':'))
        result = {
            'year': year+2000,
            'month': month,
            'day': day,
            'hour': hour,
            'minutes': minutes
        }
        return result
    
    
def get_filterd_dates(data_type):
    f_wave = open(WAVE_DATA_PATH, encoding='UTF-8')
    f_weather = open(WEATHER_DATA_PATH, encoding='UTF-8')
    f_filter = open(FILTER_DATA_PATH)

    # Skip the first line(column names)
    f_wave.readline()
    f_weather.readline()
    f_filter.readline()

    flag_wave = True
    flag_weather = True
    flag_filter = True
    while(True):
        if flag_wave:
            line_wave = f_wave.readline().strip()
            date_wave = line_wave.split(',')[0].strip()
        if flag_weather:
            line_weather = f_weather.readline().strip()
            date_weather = line_weather.split(',')[0].strip()
        if flag_filter:
            date_filter = f_filter.readline().strip()

        if not line_wave or not line_weather or not date_filter:
            break

        # Add both two lines as a new row to csv file.
        if date_wave == date_weather:
            dates_result = get_times_separately(False, date_wave)
            filter_value = get_times_separately(True, date_filter)

            cond = does_data_pass_filter(dates_result, filter_value, data_type)
            if cond == 'GOOD':
                flag_wave, flag_weather = True, True
                flag_filter = False

                result_string = '{:02d}/{:02d} {:02d}:{:02d} in [{:02d}/{:02d} {:02d}:{:02d}-{:02d}:{:02d}]'.format(
                    dates_result['month'], dates_result['day'], dates_result['hour'], dates_result['minutes'],
                    filter_value['month'], filter_value['day'], filter_value['start_hour'],
                    filter_value['start_min'], filter_value['end_hour'], filter_value['end_min'])
                yield(line_wave, line_weather, dates_result, filter_value)

            elif cond == 'NEXT_FILTER':
                flag_wave, flag_weather = False, False
                flag_filter = True
            elif cond == 'NEXT_DATA':
                flag_wave, flag_weather = True, True
                flag_filter = False

        # Read the next line of weather data only.
        elif date_wave > date_weather:
            flag_wave, flag_weather = False, True

        # Read the next line of wave data only.
        else:  # if date_wave < date_weather
            flag_wave, flag_weather = True, False

    f_wave.close()
    f_weather.close()
    f_filter.close()
    
    
def filter_condition_func(x, filter_value, dates_value):
    target = '{:2d}:{:2d}'.format(dates_value['hour'], 0)
    target_above_interval = '{:2d}:{:2d}'.format(dates_value['hour'], 10)
    target_below_interval = '{:2d}:{:2d}'.format(dates_value['hour']-1, 50)
    f_start = '{:2d}:{:2d}'.format(filter_value['start_hour'], filter_value['start_min'])
    f_end = '{:2d}:{:2d}'.format(filter_value['end_hour'], filter_value['end_min'])
    cur_hour, cur_min, cur_sec = list(map(int, x.split('.')[:-1]))
    cur_time = '{:2d}:{:2d}'.format(cur_hour, cur_min)
    
    if dates_value['hour'] == 6:
        if cur_time < target_above_interval and cur_time >= target \
        and cur_time >= f_start and cur_time <= f_end:
            return True
    else:
        if cur_time < target_above_interval and cur_time >= target_below_interval \
        and cur_time >= f_start and cur_time <= f_end:
            return True
    return False


def copy_label_img(interval=10):
    if not os.path.exists(DEST_DIR):
        os.mkdir(DEST_DIR)
    if not os.path.exists('./results'):
        os.mkdir('./results')

    with open('./results/daewoo_north_{}_result.csv'.format(DATA_TYPE), 'w') as f:
        column_names = 'file_name,year,month,day,hour,min,temperature,wind_direction,wind_speed,wave_height,wave_max_height,wave_period,wave_direction\n'
        f.write(column_names)

        cnt = 0
        for line_wave, line_weather, dates_value, filter_value in get_filterd_dates(data_type=DATA_TYPE):
            year = dates_value['year']
            month = dates_value['month']
            day = dates_value['day']
            hour = dates_value['hour']
            minutes = dates_value['minutes']

            line_weather = line_weather.split(',')
            temperature = line_weather[2].strip()
            wind_direction = line_weather[3].strip()
            wind_speed = line_weather[4].strip()
            
            line_wave = line_wave.split(',')
            wave_height = line_wave[1].strip()
            wave_max_height = line_wave[2].strip()
            wave_period = line_wave[3].strip()
            wave_direction = line_wave[4].strip()
            
            dir_name = '{}-{:02d}-{:02d}'.format(year, month, day)
            if DATA_TYPE in [2, 3]:
                dir_name += '-1'
            images = os.listdir(os.path.join(SRC_DIR, dir_name))
            
            for img in images:
                if filter_condition_func(img, filter_value, dates_value):
                    new_img_name = '{}.{}.{}.{}'.format(year, month, day, img)
                    copyfile(os.path.join(SRC_DIR, dir_name, img), os.path.join(DEST_DIR, new_img_name))
                    tmp_hour, tmp_min, tmp_sec = list(map(int, img.split('.')[:-1]))
                    row = '{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(new_img_name, year, month, day, tmp_hour, tmp_min,
                                                                            temperature, wind_direction, wind_speed,
                                                                            wave_height, wave_max_height, wave_period,
                                                                            wave_direction)
                    f.write(row)
                    cnt += 1
                    if cnt % 100 == 0:
                        print("Processing {} of {}".format(cnt, NUM_DATA))
                        
                        
if __name__ == "__main__":
    copy_label_img(interval=INTERVAL)
