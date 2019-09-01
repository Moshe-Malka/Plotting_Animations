import json
import time

from celluloid import Camera
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from Utilities import convert

######################################### Globals ##########################################################
midpointX = midpointY = Termx = Termy = patient_id = exam_id = None

_interval = 400
############################################################################################################

def collecting_drawing_data(df):
    df = df.loc[
        (df['rv']==1) &
        (df['lv']==1) &

        (df['rx'] < (midpointX + 1000)) &
        (df['ry'] < (midpointY + 400)) &

        (df['rx'] > (midpointX - 1000)) &
        (df['ry'] > (midpointY - 400)) &

        (df['lx'] < (midpointX + 1000)) &
        (df['ly'] < (midpointY + 400)) &

        (df['lx'] > (midpointX - 1000)) &
        (df['ly'] > (midpointY - 400))
        ]

    df = df.loc[df['TestType'] == 'Starbismus', [
    'RightEye', 'LeftEye', 'lx', 'ly', 'rx', 'ry']]
    
    if df.empty: return [pd.DataFrame(), pd.DataFrame()]
    
    up_case = df.loc[(df['RightEye'] == 1), ['lx', 'ly', 'rx', 'ry']]
    up_case = normilize_y(up_case.copy())

    down_case = df.loc[(df['LeftEye'] == 1), ['lx', 'ly', 'rx', 'ry']]
    down_case = normilize_y(down_case.copy())

    up_case, down_case = remove_outliers(up_case, down_case)

    return [up_case, down_case]

def draw(down_case, up_case, to_plot):
    for p in to_plot:
        fig = plt.figure()
        camera = Camera(fig)
        if p['loc'] == 'up':
            x = up_case[p['x']].values
            y = up_case[p['y']].values
        else:
            x = down_case[p['x']].values
            y = down_case[p['y']].values
            
        if len(x) > 60:
            case_range = int(len(x) / 6)
        elif 60 >= len(x) >= 30:
            case_range = int(len(x) / 4)
        else: # <30
            case_range = 5

        # ***************** IF WE WANT ORIGINAL SCALE *****************
        # plt.axis([0.0, 1368, 0.0, 912])

        # ***************** IF WE WANT DYNAMIC SCALE *****************
        plt.axis(xmin=min(x)-200, xmax=max(x)+200, ymin=min(y)-200, ymax=max(y)+200)
        plt.title(f"{p['loc'].title()} - {p['text']} [{len(x)}]")

        di = 255 / len(x)
        colormap = [(0 + x * di / 256, .1, 1.0 - x * di / 256) for x in range(len(x))]
        for i in range(len(x)):
            plt.plot(x[0], y[0], marker='o', color='green')  # the start (first point)
            plt.plot(x[len(x)-1], y[len(y)-1], marker='o', color='c')  # the end  (last point)
            plt.plot(np.mean(x), np.mean(y), marker='o', color='yellow') # total data average point

            plt.plot(midpointX, midpointY, marker='+', color='k')  # the middle (black)
            plt.plot([midpointX-77, midpointX-77], [midpointY-72, midpointY+72], "k-", linewidth=0.8) # center black cube - left
            plt.plot([midpointX+77, midpointX+77], [midpointY-72, midpointY+72], "k-", linewidth=0.8) # center black cube - right
            plt.plot([midpointX-77, midpointX+77], [midpointY-72, midpointY-72], "k-", linewidth=0.8) # center black cube - bottom
            plt.plot([midpointX-77, midpointX+77], [midpointY+72, midpointY+72], "k-", linewidth=0.8) # center black cube - top
            
            plt.plot(np.mean(x[:case_range]), np.mean(y[:case_range]), marker='X', color='xkcd:fluorescent green', markeredgewidth=0.5, markeredgecolor='b')  # the start (first 10)
            plt.plot(np.mean(x[-case_range:]), np.mean(y[-case_range:]), marker='X', color='xkcd:red orange', markeredgewidth=0.5, markeredgecolor='b')  # the end (last 10)
            
            plt.scatter(x[:i+1], y[:i+1], marker='o', color=colormap[:i+1])
            
            camera.snap()
            
        animation = camera.animate(interval=_interval, repeat=False)
        animation.save(f'strabismus_{patient_id}_{exam_id}_{p["loc"]}_{p["text"]}_{exam_id}.gif', writer='pillow')

        plt.close(fig)
        print(f"Finished Animating Strabismus Plot for : {p['loc']} , {p['text']}")

def normilize_y(data):
    """ turns the y axis upside down.
    Args:
            data (DataFrame): the data we want to change. [lx,ly,rx,ry]
    Returns:
            DataFrame: the flipped data
    """
    data['ly'] = Termy - data['ly']
    data['ry'] = Termy - data['ry']
    return data

def get_teller_data(df):
	""" get all of the teller data after adjusting it's columns (lx, lt, rx, ry). """
	df = df.loc[
			(df['rv'] == 1) &
			(df['lv'] == 1) &
            (df['rx'] > -1368) &
            (df['ry'] > -611) &
            (df['lx'] > -1368) &
            (df['ly'] > -611) &
            (df['rx'] < 1368) &
            (df['ry'] < 1000) &
            (df['lx'] < 1368) &
            (df['ly'] < 1000)
			]
	df = df.loc[df['Index'] > 4, ['Index', 'TellerType', 'RightEye', 'LeftEye', 'X', 'Y', 'Width', 'Height', 'lx', 'ly', 'rx', 'ry', 'rv', 'lv']]
	return df

def get_params(tellers):
    teller_indexs = tellers.loc[tellers['TellerType'] != 0]['Index'].unique().tolist()
    teller_pre_indexs = [x-1 for x in teller_indexs]
    pairs = list(zip(teller_indexs, teller_pre_indexs))
    for pair in pairs:
        teller_data = tellers.loc[
            tellers.Index == pair[0],
            ['TellerType', 'RightEye', 'LeftEye', 'Index', 'X', 'Y', 'Width', 'Height', 'lx', 'ly', 'rx', 'ry', 'rv', 'lv']
        ]
        pre_data = tellers.loc[
            tellers.Index == pair[1],
            ['TellerType', 'RightEye', 'LeftEye', 'Index', 'X', 'Y', 'Width', 'Height', 'lx', 'ly', 'rx', 'ry', 'rv', 'lv']
        ]
        teller_type = teller_data.TellerType._values[0]
        opened_eye = "R" if teller_data.RightEye.values.max() == True else "L"
        plot_teller(teller_data, pre_data, teller_type, opened_eye)

def plot_teller(teller_data, pre_data, teller_type, opened_eye):
    eye = opened_eye.lower()
    center_x = teller_data['X'].iloc[0]   	# center of teller picture on the X scale
    center_y = teller_data['Y'].iloc[0]   	# center of the teller picture on the Y scale
    width = teller_data['Width'].iloc[0]	# width of inner box
    height = teller_data['Height'].iloc[0]	# height of inner box
    teller_position = "left" if (center_x < 500) else "right"
    ind = teller_data['Index'].iloc[0]
    
    x = teller_data[f'{eye}x'].values
    y = teller_data[f'{eye}y'].values

    # last 20 points before teller.
    pre_data_subset_x = pre_data[f'{eye}x'].iloc[-30:].values
    pre_data_subset_y = pre_data[f'{eye}y'].iloc[-30:].values

    fig = plt.figure()
    camera = Camera(fig)

    if len(x) > 60:
        case_range = int(len(x) / 6)
    elif 60 >= len(x) >= 30:
        case_range = int(len(x) / 4)
    else: # <30
        case_range = 5

    # Setting custom ticks
    # plt.xticks(np.arange(0, max(x)+50, step=50))
    # plt.yticks(np.arange(0, max(y)+50, step=50))

    # ***************** IF WE WANT DYNAMIC SCALE - BY VALUES *****************
    tmp_x_values = np.concatenate((x, pre_data_subset_x))
    tmp_y_values = np.concatenate((y, pre_data_subset_y))
    zoom_parameter = 150

    # ***************** IF WE WANT DYNAMIC SCALE - BY SCREEN WIDTH/HEIGHT *****************
    # plt.xlim(0, Termx)
    # plt.ylim(0, Termy)

    # ***************** IF WE WANT ORIGINAL SCALE *****************
    # plt.xlim(0, 1368)
    # plt.ylim(0, 912)

    di = 255 / len(x)
    colormap = [(0 + x * di / 256, .1, 1.0 - x * di / 256) for x in range(len(x))]
    # colormap = [ (x, .1, y) for x, y in zip(np.arange(1, 0, -1 / len(x)), np.arange(0, 1, 1 / len(x)))]
    
    for j in range(len(pre_data_subset_x)):
        plt.scatter(pre_data_subset_x[:j+1], pre_data_subset_y[:j+1], marker='s', color='orange')
        plt.plot(center_x, center_y, marker='o', color='k')     # center of teller
        
        # Constructing the teller bound and it's outer box.
        if teller_position == "left":
            plt.plot([int(center_x - width), int(center_x - width)], [int(center_y - height), int(center_y + height)], "k-")        # outer box - left
            plt.plot([int(center_x + width / 1.5), int(center_x + width / 1.5)], [int(center_y - height), int(center_y + height)], "k-")        # outer box - right
            plt.plot([int(center_x + width), int(center_x + width)], [int(center_y - height), int(center_y + height)], "k-", linestyle='dashed', linewidth=1.2)        # outer box - right - dashed
            plt.plot([int(center_x - width), int(center_x + width / 1.5)], [int(center_y + height), int(center_y + height)], "k-")        # outer box - top
            plt.plot([int(center_x - width), int(center_x + width / 1.5)], [int(center_y - height), int(center_y - height)], "k-")        # outer box - bottom
            plt.plot([int(center_x - width), int(center_x + width)], [int(center_y + height), int(center_y + height)], "k-", linestyle='dashed', linewidth=1.2)        # outer box - top - dashed
            plt.plot([int(center_x - width), int(center_x + width)], [int(center_y - height), int(center_y - height)], "k-", linestyle='dashed', linewidth=1.2)        # outer box - bottom - dashed
        else:
            plt.plot([int(center_x - width / 1.5), int(center_x - width / 1.5)], [int(center_y - height), int(center_y + height)], "k-")        # outer box - left
            plt.plot([int(center_x - width), int(center_x - width)], [int(center_y - height), int(center_y + height)], "k-", linestyle='dashed', linewidth=1.2)        # outer box - left - dashed
            plt.plot([int(center_x + width), int(center_x + width)], [int(center_y - height), int(center_y + height)], "k-")        # outer box - right
            plt.plot([int(center_x - width / 1.5), int(center_x + width)], [int(center_y + height), int(center_y + height)], "k-")        # outer box - top
            plt.plot([int(center_x - width / 1.5), int(center_x + width)], [int(center_y - height), int(center_y - height)], "k-")        # outer box - bottom
            plt.plot([int(center_x - width), int(center_x + width)], [int(center_y + height), int(center_y + height)], "k-", linestyle='dashed', linewidth=1.2)        # outer box - top - dashed
            plt.plot([int(center_x - width), int(center_x + width)], [int(center_y - height), int(center_y - height)], "k-", linestyle='dashed', linewidth=1.2)        # outer box - bottom - dashed

        plt.plot([int(center_x - width/2), int(center_x - width/2)], [int(center_y - height/2), int(center_y + height/2)], "k-")        # inner box - left
        plt.plot([int(center_x + width/2), int(center_x + width/2)], [int(center_y - height/2), int(center_y + height/2)], "k-")        # inner box - right
        plt.plot([int(center_x - width/2), int(center_x + width/2)], [int(center_y + height/2), int(center_y + height/2)], "k-")        # inner box - top
        plt.plot([int(center_x - width/2), int(center_x + width/2)], [int(center_y - height/2), int(center_y - height/2)], "k-")        # inner box - bottom
        
        camera.snap()

    for i in range(len(x)):
        plt.plot(np.mean(x), np.mean(y), marker='o', color='y') # mean point of whole data.
        plt.plot(x[0], y[0], marker='o', color='green')  # the first point.
        plt.plot(x[len(x) - 1], y[len(y) - 1], marker='o', color="c")  # the last point.

        plt.plot(center_x, center_y, marker='o', color='k')     # center of teller
        
        # Constructing the teller bound and it's outer box.
        if teller_position == "left":
            plt.plot([int(center_x - width), int(center_x - width)], [int(center_y - height), int(center_y + height)], "k-")        # outer box - left
            plt.plot([int(center_x + width / 1.5), int(center_x + width / 1.5)], [int(center_y - height), int(center_y + height)], "k-")        # outer box - right
            plt.plot([int(center_x + width), int(center_x + width)], [int(center_y - height), int(center_y + height)], "k-", linestyle='dashed', linewidth=1.2)        # outer box - right - dashed
            plt.plot([int(center_x - width), int(center_x + width / 1.5)], [int(center_y + height), int(center_y + height)], "k-")        # outer box - top
            plt.plot([int(center_x - width), int(center_x + width / 1.5)], [int(center_y - height), int(center_y - height)], "k-")        # outer box - bottom
            plt.plot([int(center_x - width), int(center_x + width)], [int(center_y + height), int(center_y + height)], "k-", linestyle='dashed', linewidth=1.2)        # outer box - top - dashed
            plt.plot([int(center_x - width), int(center_x + width)], [int(center_y - height), int(center_y - height)], "k-", linestyle='dashed', linewidth=1.2)        # outer box - bottom - dashed
        else:
            plt.plot([int(center_x - width / 1.5), int(center_x - width / 1.5)], [int(center_y - height), int(center_y + height)], "k-")        # outer box - left
            plt.plot([int(center_x - width), int(center_x - width)], [int(center_y - height), int(center_y + height)], "k-", linestyle='dashed', linewidth=1.2)        # outer box - left - dashed
            plt.plot([int(center_x + width), int(center_x + width)], [int(center_y - height), int(center_y + height)], "k-")        # outer box - right
            plt.plot([int(center_x - width / 1.5), int(center_x + width)], [int(center_y + height), int(center_y + height)], "k-")        # outer box - top
            plt.plot([int(center_x - width / 1.5), int(center_x + width)], [int(center_y - height), int(center_y - height)], "k-")        # outer box - bottom
            plt.plot([int(center_x - width), int(center_x + width)], [int(center_y + height), int(center_y + height)], "k-", linestyle='dashed', linewidth=1.2)        # outer box - top - dashed
            plt.plot([int(center_x - width), int(center_x + width)], [int(center_y - height), int(center_y - height)], "k-", linestyle='dashed', linewidth=1.2)        # outer box - bottom - dashed

        plt.plot([int(center_x - width/2), int(center_x - width/2)], [int(center_y - height/2), int(center_y + height/2)], "k-")        # inner box - left
        plt.plot([int(center_x + width/2), int(center_x + width/2)], [int(center_y - height/2), int(center_y + height/2)], "k-")        # inner box - right
        plt.plot([int(center_x - width/2), int(center_x + width/2)], [int(center_y + height/2), int(center_y + height/2)], "k-")        # inner box - top
        plt.plot([int(center_x - width/2), int(center_x + width/2)], [int(center_y - height/2), int(center_y - height/2)], "k-")        # inner box - bottom

        plt.plot(np.mean(x[:case_range]), np.mean(y[:case_range]), marker='X', color='xkcd:fluorescent green', markeredgewidth=0.5, markeredgecolor='b')  # the start (first 10)
        plt.plot(np.mean(x[-case_range:]), np.mean(y[-case_range:]), marker='X', color='xkcd:red orange', markeredgewidth=0.5, markeredgecolor='b')  # the end (last 10)

        plt.title(f"Patient Id: {patient_id} Total Points: {len(x)}\nTeller: {teller_type} - Eye: {eye.upper()} - Teller Position: {'R' if teller_position == 'right' else 'L'} - Index: {ind}")
        plt.text(x[i]+5, y[i]+5, f'{i+1}', size='13' ,color='white', style='italic', bbox={'boxstyle':'round', 'facecolor': 'gray', 'alpha': 0.8})
        
        plt.scatter(x[:i+1], y[:i+1], marker='o', color=colormap[:i+1], label=f"{i+1}")
        
        plt.xlim(min(tmp_x_values) - zoom_parameter, max(tmp_x_values) + zoom_parameter)
        plt.ylim(min(tmp_y_values) - zoom_parameter, max(tmp_y_values) + zoom_parameter)

        camera.snap()

    animation = camera.animate(interval=_interval, repeat=False)

    animation.save(f'tellers_{patient_id}_{exam_id}_{teller_type}_{ind}_{opened_eye}_{teller_position}.gif', writer='pillow')

    plt.close(fig)
    print(f"Finished Animating Teller Plot for : Patient Id : {patient_id}, Exam Id : {exam_id}, Teller Type : {teller_type}, Teller Index : {ind}, Opened Eye : {opened_eye}, Teller Position : {teller_position}")

def remove_outliers(up, down):
    """ drops the points that are farthest from the main mass."""
    up_drop = set()
    down_drop = set()
    
    for column in up.columns[:4]:
        center_of_mass = int(up[column].mean())
        values = np.abs(center_of_mass - up[column])
        if values[values > 150].count() <= 5:
            outliers = values[values > 150].index.tolist()
            up_drop.update(outliers)

    for column in down.columns[:4]: 
        center_of_mass = int(down[column].mean())
        values = np.abs(center_of_mass - down[column])
        if values[values > 150].count() <= 5:
            outliers = values[values > 150].index.tolist()
            down_drop.update(outliers)
    
    return [up.drop(up_drop), down.drop(down_drop)]

def plotting(strabismus=True, tellers=True):
    global patient_id, exam_id, Termx, Termy, midpointX, midpointY
    with open('tmp.json', 'rb') as j:
        input_json = json.loads(j.read())

    print(f"Protocol : {input_json['Protocol']}")

    patient_id = input_json.get("PatientId", 'No Patient Id')
    exam_id = input_json.get("ExamId", 'No Exam Id')

    df = convert(input_json)

    Termx = input_json.get("ScreenWidth", 684)
    Termy = input_json.get("ScreenHight", 456)
    midpointX = int(Termx / 2)
    midpointY = int(Termy / 2)

    if(df.empty):
        print("NO DATA : number of rows (data points) for left/right/both eyes is lower than 8 !")
        return

    if strabismus:
        strabismus_to_plot = [
            {
                'loc' : 'up',
                'x' : 'lx',
                'y' : 'ly',
                'text' : 'L Eye - closed'
            },
            {
                'loc' : 'up',
                'x' : 'rx',
                'y' : 'ry',
                'text' : 'R Eye - opened'
            },
            {
                'loc' : 'down',
                'x' : 'lx',
                'y' : 'ly',
                'text' : 'L Eye - opened'
            },
            {
                'loc' : 'down',
                'x' : 'rx',
                'y' : 'ry',
                'text' : 'R Eye - closed'
            }
        ]
        start_time_strabismus = time.time()

        # with removing outliers
        up_case_draw, down_case_draw = collecting_drawing_data(df)
        if up_case_draw.empty or down_case_draw.empty: print("Empty Strabismus DataFrame !")
        else: draw(down_case_draw, up_case_draw, strabismus_to_plot)

        print(f"Strabismus --- {(time.time() - start_time_strabismus)} seconds ---")
    
    if tellers:
        start_time_tellers = time.time()
        teller_df = get_teller_data(df)
        if teller_df.empty: print("Empty Tellers DataFrame !")
        else: get_params(teller_df)
        print(f"Tellers --- {(time.time() - start_time_tellers)} seconds ---")

if __name__ == '__main__':
    plotting()