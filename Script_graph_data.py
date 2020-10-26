from os import chdir, listdir, mkdir, remove
from urllib.request import urlretrieve
from tkinter import font as tkFont
from datetime import datetime, timedelta, date
import pathlib, time, tkinter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.animation as animation
from matplotlib import ticker, rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable



Root_directory = str(pathlib.Path(__file__).parent.absolute()) # Strings with path of different directories
Datafiles_directory = Root_directory + '\\Datafiles'
Animation_directory = Root_directory + '\\Animations'

for Directory in [Datafiles_directory, Animation_directory]: # Create necessary directories if they don't exist
    try:
        mkdir(Directory)
    except:
        pass

Title_fontsize = 25 # Size in pixels of different elements in the graph
Date_fontsize = 25
Annotation_fontsize = 15
Axis_label_fontsize = 20
Axis_tick_fontsize = 15

Marker_ray = 8 # Ray in pixels of the markers in the scater graph

Annotation_offset = - Annotation_fontsize * 0.05 # Correction of the spacing between points in the graph and their annotations

Title_pad = Title_fontsize * 0.5 # Spacing betwenn title and plotting area
Axis_label_pad = Axis_label_fontsize * 0.5 # Spacing betwenn axis label and plotting area


Animation_interval = 200 # Interval in milliseconds between each frame in animation
Animation_fps = int(1/Animation_interval * 1e3) # Frames per second of the animation


Graph_font = rcParams['font.family'] # Tkinter font object that will be used to calculate the area occupied by each annotation in Annotations_frame()
tkinter.Frame().destroy()
font = tkFont.Font(family = Graph_font, size = Annotation_fontsize)


def Main_script(X_axis_inc = 1, Y_axis_inc = 7, Z_axis_inc = 12, Date_start = None, Date_end = None):
    """
    Main routine to execute to download, extract, reconstruct and plot COVID data
    
    Parameters:
        - X_axis_inc: Integer, data to use for the X axis. Default is 1 (Total cases per million)
        - Y_axis_inc: Integer, data to use for the Y axis. Default is 7 (Total deaths per million)
        - Z_axis_inc: Integer, data to use for the colors of the points. Default is 12 (Positivity rate)
        - Date_start: String, starting date of the animation
        - Date_end: String, ending date of the animation
    Date are None by default to use every available date in the data (see first lines of this fuction)
    
    Returns:
        ani: Matplotlib animation object. If it's not returned, the animation isn't displayed
    
    Axis incs:
        0  - Total cases
        1  - Total cases per million
        2  - New cases
        3  - New cases smoothed
        4  - New cases per million
        5  - New cases per million smoothed
        6  - Total deaths
        7  - Total deaths per million
        8  - New deaths
        9  - New deaths smoothed
        10 - New deaths per million
        11 - New deaths per million smoothed
        12 - Positivity rate
        13 - Testing policy
        
    """
    Timer_start = time.perf_counter()
    
    print('Collecting data from Our World in Data')
    COVID_data, Date_start_raw_data, Date_end_raw_data = Extract_data() # Download and extract raw COVID data
    
    if Date_start == None: # If no start date is specified, it is set to the first date in the data. The end date is then set to the last date in the data. That way, we can display the whole data without having to know when it starts and ends
        Date_start = Date_start_raw_data
        Date_end = Date_end_raw_data
    
    elif Date_end == None: Date_end = Date_start # But if no end date is specified, only the start date is displayed
    
    print('Recontructing missing chunks in the data by linear interplolation')
    COVID_data_reconstructed = Reconstruct_COVID_data(COVID_data) # Reconstruct the missing data
    
    print('Exporting data in files')
    Export_in_files(COVID_data, COVID_data_reconstructed) # Export the original and reconstructed data in CSV files, just to have them and be able to look whenever we want
    
    print('Isolating data to plot')
    COVID_data_scatter = Extract_data_for_plotting(COVID_data_reconstructed, X_axis_inc, Y_axis_inc, Z_axis_inc, Date_start, Date_end) # Filter data to only keep the axes we want to plot
    
    print('Plotting data')
    ani, COVID_data_scatter_names = Scatter_graph(COVID_data_scatter) # Plot the data
    
    print('Exporting animation as video')
    Writer = animation.writers['ffmpeg'] # Export the file
    writer = Writer(fps = Animation_fps, metadata=dict(artist='Me'), bitrate=1800)
    
    Annimation_file = Animation_directory + '\\%s vs %s with %s from %s to %s.mp4' % (tuple(COVID_data_scatter_names) + (Date_start, Date_end))
    ani.save(Annimation_file, writer = writer)
    
    print('\nProcessing done in %0.2f minutes' % ((time.perf_counter() - Timer_start) / 60))
    return ani
    


def Extract_data():
    """
    Extracts and formats data in dictionnaries from Our World in Data CSV files
    
    Parameters: Nothing
    
    Returns:
        - COVID_data: Dictionnary of cases, deaths and positivity rate data throughout the world
        - Population_data: Dictionnary of population for each country
    """
    chdir(Datafiles_directory) # Empty the datafiles directory
    File_list = listdir()
    
    for File in File_list:
        remove(File)
            
    COVID_data_path = Datafiles_directory + '\\OWID COVID data %s.csv' % (date.today().isoformat()) # String with path of COVID data (where it will be stored when downloaded)
    
    urlretrieve('https://raw.githubusercontent.com/owid/COVID-19-data/master/public/data/owid-covid-data.csv', COVID_data_path) # Download and extract the data
    COVID_data_file = open(COVID_data_path, 'r')

    COVID_raw_data = COVID_data_file.readlines()
    COVID_raw_data = [Row.split(',') for Row in COVID_raw_data[1:]]

    COVID_data = {'_Country': {'Date': ['Total cases', 'Total cases per million', 'New cases', 'New cases smoothed', 'New cases per million', 'New cases per million smoothed', 'Total deaths', 'Total deaths per million', 'New deaths', 'New deaths smoothed', 'New deaths per million', 'New deaths per million smoothed', 'Positivity rate', 'Testing policy']}}
    Date_list = []
    
    for Row_inc in range(len(COVID_raw_data)): # For each row in the file...
        Country = COVID_raw_data[Row_inc][2]
        Date = COVID_raw_data[Row_inc][3]
        
        if COVID_raw_data[Row_inc][2] not in COVID_data: COVID_data[Country] = {} # If a new country is encountered, a new entry to the dictionnary COVID_data is added
        
        if Date not in Date_list: Date_list.append(Date) # If a new date is encoutered, it is added to the corresponding list
        
        COVID_data[Country][Date] = []
        for Column_inc in [4, 10, 5, 6, 11, 12, 7, 13, 8, 9, 14, 15, 23, 24]: # For each column we want to extract...
            Data_item = COVID_raw_data[Row_inc][Column_inc]
            if Column_inc != 24: # Column_inc of 24 is the testing policy and is a string so can't appended as a float, prompting this exception
                if Data_item == '': COVID_data[Country][Date].append(None) # If there's nothing, a None element is added
                else: COVID_data[Country][Date].append(float(COVID_raw_data[Row_inc][Column_inc]))
        
            else: COVID_data[Country][Date].append(COVID_raw_data[Row_inc][Column_inc])
                
        if COVID_raw_data[Row_inc][2] == 'International' or COVID_raw_data[Row_inc][2] == 'World': # The entries "World" and "International" aren't interesting so they are ignored
            break
        
        COVID_data_file.close()

    Date_start_raw_data, Date_end_raw_data = min(Date_list), max(Date_list)

    return COVID_data, Date_start_raw_data, Date_end_raw_data



def Reconstruct_COVID_data(COVID_data):
    """
    Reconstructs missing chunks of data by linear interpolation
    
    Parameters:
        COVID_data: Dictionnary of cases, deaths and positivity rates data throughout the world
    
    Returns:
        COVID_data_reconstructed: Reconstructed dictionnary of cases, deaths and positivity rates data throughout the world
    """
    COVID_data_reconstructed = {}
    COVID_data_reconstructed['_Country'] = COVID_data['_Country']
    
    Countries_list = list(COVID_data.keys())[1:]
    
    for Country in Countries_list: # For each country...
        COVID_data_single_country = list(COVID_data[Country].values()) # Extract the matrix containing the data and transpose it. That way, each element of a single list in the array corresponds to one column (see help of Main_script) and it makes it easier to navigate through each column and recontruct the missing elements
        T_COVID_data_single_country = list(map(list, zip(*COVID_data_single_country)))
        
        for Column_inc in range(len(T_COVID_data_single_country)): # For each column...
            Column = T_COVID_data_single_country[Column_inc]
            Max_column_inc = len(Column) - 1
            
            Row_inc = 0
            while Column[Row_inc] == None and Row_inc < Max_column_inc: # Recontructing missing data at the beginning is impossible so we just skip the first rows with a None in them
                Row_inc += 1
                
            
            if None in Column: # If a None is in the list (meaning there are bits of data missing)...
                while Row_inc < Max_column_inc: # Not including this line could prompt an index error
                    if Column[Row_inc] == None: # When a None in encoutered...
                        None_interval_start = Row_inc # Recording when the segments of None starts and ends
        
                        while Column[Row_inc] == None and Row_inc < Max_column_inc:
                            Row_inc += 1
                        
                        None_interval_end = Row_inc - 1
                        Interpolation_interval_length = None_interval_end - None_interval_start + 2
                        
                        if Row_inc < Max_column_inc: # Reconstruction of the segment by linear interpolation : Y = mX + b with m = (Y_max - Y_min) / (X_max - X_min)
                            m = (Column[None_interval_end + 1] - Column[None_interval_start - 1]) / Interpolation_interval_length
                            b = Column[None_interval_start - 1]
                            
                            for Row_inc in range(None_interval_start, Row_inc):
                                T_COVID_data_single_country[Column_inc][Row_inc] = m * (Row_inc - None_interval_start + 1) + b
                        
                        else: # In the case the None segment goes on until the end, the last known value is just copied
                            for Row_inc in range(None_interval_start, Row_inc + 1):
                                T_COVID_data_single_country[Column_inc][Row_inc] = T_COVID_data_single_country[Column_inc][None_interval_start - 1]
                        
                    Row_inc += 1
    
        COVID_data_single_country_reconstructed = list(map(list, zip(*T_COVID_data_single_country))) # Retranspose the matrix to get the reconstruted data in the correct format
        
        Date_list_country = list(COVID_data[Country].keys()) # Add the reconstructed data to the appropriate dictionnary
        COVID_data_reconstructed[Country] = {}
        for Date_inc in range(len(Date_list_country)):
            Date = Date_list_country[Date_inc]
            COVID_data_reconstructed[Country][Date] = COVID_data_single_country_reconstructed[Date_inc]
    
    return COVID_data_reconstructed



def Export_in_files(COVID_data, COVID_data_reconstructed):
    """
    Exports the raw and reconstructed data in seperate files
    
    Parameters:
        - COVID_data: Dictionnary of cases, deaths and positivity rate data throughout the world
        - Covid_data_reconstructued: Reconstructed dictionnary of cases, deaths and positivity rates data throughout the world
    
    Returns: Nothing
    """
    F_data_file = open(Datafiles_directory + '\\OWID COVID data %s formatted.csv' % (date.today().isoformat()), 'w')
    FR_data_file = open(Datafiles_directory + '\\OWID COVID data %s formatted reconstructed.csv' % (date.today().isoformat()), 'w')
    
    COVID_data_lists = [COVID_data, COVID_data_reconstructed]
    Data_file_list = [F_data_file, FR_data_file]
    Countries_list = list(COVID_data.keys())[1:]
    
    for Data_set_inc in range(2): # Each data list (raw and reconstructed) is written in its corresponding file
        COVID_data_temp = COVID_data_lists[Data_set_inc]
        Data_file_temp = Data_file_list[Data_set_inc]
        
        Data_file_temp.write('Country;Date;' + ';'.join(COVID_data_temp['_Country']['Date']) + '\n')
        
        for Country in Countries_list:
            COVID_data_single_country = COVID_data_temp[Country]
            
            Date_list = list(COVID_data[Country].keys())
            for Date in Date_list:
                COVID_data_single_country_single_date = COVID_data_single_country[Date]
                Row_reformatted = ['' if Item == None else str(Item).replace('.', ',') for Item in COVID_data_single_country_single_date] # None elements are replaced by empty strings because an empty cell is better to see that there is no data in excel rather than None
                
                Data_file_temp.write('%s;%s;' % (Country, Date))
                Data_file_temp.write(';'.join(str(Item) for Item in Row_reformatted))
                Data_file_temp.write('\n')
                
        Data_file_temp.close()



def Extract_data_for_plotting(COVID_data, X_Axis_inc, Y_Axis_inc, Z_Axis_inc, Date_start, Date_end, Keep_no_PR = True):
    """
    Extract data from recontructed COVID data in order to only keep data that will be plotted
    
    Parameters:
        - COVID_data: Dictionnary of cases, deaths and positivity rates data throughout the world (usually reconstructed)
        - X_axis_inc: Integer, data to use for the X axis
        - Y_axis_inc: Integer, data to use for the Y axis
        - Z_axis_inc: Integer, data to use for the colors of the points
        - Date_start: String, starting date of the animation
        - Date_end: String, ending date of the animation
        - Keep_no_PR: Boolean indicating whether or not countries without a positivity rate have to be kept. Default is True
    
    Returns:
        COVID_data_scatter: Reconstructed dictionnary of the 3 columns the user asked to plot throughout time
    """
    Date_start_obj = datetime.strptime(Date_start, '%Y-%m-%d') # Create a list of all the dates to extract
    Date_end_obj = datetime.strptime(Date_end, '%Y-%m-%d')
    Date_difference = (Date_end_obj - Date_start_obj).days + 1
    
    Date_list = [(Date_start_obj + timedelta(Days)).isoformat()[:10] for Days in range(Date_difference)]
    
    Countries_list = list(COVID_data.keys())[1:]

    COVID_data_scatter = {'0Date': {'Country': [COVID_data['_Country']['Date'][Axis_inc] for Axis_inc in [X_Axis_inc, Y_Axis_inc, Z_Axis_inc]]}}
    
    for Date in Date_list: # For each date and each country...
        COVID_data_scatter[Date] = {}
        for Country in Countries_list:
            try:
                Data_items = [COVID_data[Country][Date][Axis_inc] for Axis_inc in [X_Axis_inc, Y_Axis_inc, Z_Axis_inc]] # This line will prompt an error in case the data doesn't exist, hence the try - except structure (much easier than 10 000 conditions to try to figure out if the data exists for a date and country)
                
                if None not in Data_items[:2] and not (not Keep_no_PR and Data_items[2] == None): # Any data point that has a None as its X or Y coordinate is exlcuded, and also Z if asked by the user
                    if min(Data_items[:2]) > 0: COVID_data_scatter[Date][Country] = Data_items # Since the graph is in logscale, points with 0 as their X or Y coordinate are excluded (because log(0) doesn't exist).
                    # This double verification can't be done in one line because having None in a list you're trying to find the minimum of prompts an error
            except: pass
        
        if COVID_data_scatter[Date] == {}: COVID_data_scatter.pop(Date)
    
    return COVID_data_scatter



def Annotations_frame(Points_to_display, Countries_displayed, Frame_limits):
    """
    Tells which countries to annotate and which not to. Since the lists in parameters are sorted by descending order of positivity rate, the countries with higher positivity rates will be examined first and thus annotatd with more priority
    
    Parameters:
        - Points_to_display: List of X and Y coordinates of each point displayed on the graph
        - Countries_displayed: List of countries displayed on the graph
        - Frame_limits: Tuple, limits of the plotting area (X_min, X_max, Y_min, Y_max)
    
    Returns:
        - Countries_to_annotate: List of countries to annotate
        - Annotations_mask: Numpy array of bools, outline of the annotations. This variable is only used in this function to decide which countries to annotate and which not to but I had so many problems in finding the correct formulas that just in case, I wanted to be able to display it easilly in Scatter_graph() even after solving all problems
    """
    X_list_frame, Y_list_frame = zip(*Points_to_display) # Transform tuples of (X, Y) into 2 distinct lists of X and Y coordinates
    
    Frame_limits_log = list(map(np.log10, Frame_limits))
    X_min_log, X_max_log, Y_min_log, Y_max_log = Frame_limits_log
    
    fig = plt.gcf()
    ax = plt.gca()
    
    ax_bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted()) # Get size in pixels of the plotting area
    ax_width, ax_height = ax_bbox.width, ax_bbox.height
    ax_width *= fig.dpi
    ax_height *= fig.dpi
    
    Annotations_mask = np.zeros((int(ax_width), int(ax_height)), bool) # Array of bools same size as plotting area where outlines of annotations will be stored
    
    Countries_to_annotate = {}
    
    for Country_inc in range(len(Countries_displayed)): # For each country...
        Country = Countries_displayed[Country_inc]
        
        Annotation_width_enlargment = 1.3 # Slight corrections to make the annotation outlines fit as best as possible. Found by trial and error
        Annotation_height_enlargment = 1.6
        
        Label_size = 0.5 * np.array([font.measure(Country)*Annotation_width_enlargment, Annotation_fontsize * Annotation_height_enlargment]) # Everything is divided by 2 because the origin of the anotation outline is in its center
        Offset = [0, Marker_ray + Annotation_fontsize/72*fig.dpi*0.7 + Annotation_offset] # Distance between point and annotation. Annotation_fontsize is in points so it has to be converted to pixels (1 inch = 72 points = screen dpi). 0.56 is just a correction found by trial and error
        
        Country = Countries_displayed[Country_inc]
        Country_coords = Points_to_display[Country_inc]
        
        List_slice = [] # Get indices delimiting the outline of the annotation in the plotting area
        for Axis_inc in range(2):
            Min_log, Max_log = Frame_limits_log[Axis_inc*2 : Axis_inc*2 + 2] # Simple transformation: Y = (Y_max - Y_min) / (X_max - X_min) * (X - X_min) + Y_min
            Coodrs_transformation = lambda x: (Annotations_mask.shape[Axis_inc] - 1)/(Max_log - Min_log) * (np.log10(x) - Min_log)
            
            for Label_offset_sign in range(-1, 2, 2):
                List_slice.append(sum([Coodrs_transformation(Country_coords[Axis_inc]), Offset[Axis_inc], Label_offset_sign * Label_size[Axis_inc]]))
        
        Slice_X_min, Slice_X_max, Slice_Y_min, Slice_Y_max = map(int, List_slice)
        Annotation_slice = np.s_[Slice_X_min : Slice_X_max + 1, Slice_Y_min : Slice_Y_max + 1]
         
        if not np.any(Annotations_mask[Annotation_slice]): # If there isn't a True in the current annotation outline (meaing there already is another annotation displayed)...
            Countries_to_annotate[Country] = Points_to_display[Country_inc] # The country has to be annotated
            Annotations_mask[Annotation_slice] = True # All the elements in Annotations_mask in this area are set to True to signify there is now an annotation displayed there
    
    return Countries_to_annotate, Annotations_mask
    


def Scatter_graph(COVID_data_scatter, Display_annotations_mask = False):
    """
    Plots data entered in parameters
    
    Parameters:
        - COVID_data_scatter: Reconstructed dictionnary of the 3 columns the user asked to plot throughout time
        - Display_annotations_mask: Boolean indicating whether to display the outline of annotations created by Annotations_frame() or not
    
    Returns:
        - ani: Animation object created by matplotlib
        - COVID_data_scatter_names: List of names of the columns plotted
    """
    COVID_data_scatter_names = COVID_data_scatter.pop('0Date')['Country'] # Extract names of columns plotted
    
    X_axis, Y_axis, Z_axis = [], [], [] # Separate the axes in COVID_data_scatter in order to find the minimum and maximum along each axis
    for Date_item in COVID_data_scatter.values():
        for Country_item in Date_item.values():
            for Axis_inc in range(3):
                [X_axis, Y_axis, Z_axis][Axis_inc].append(Country_item[Axis_inc])
    
    Min_list, Max_list = [], [] # Limits of the plotting area
    Graph_window_margin = 2 # Since the graph is in log scale, the plotting area can't be extended using  New max = Factor * (Max - Min) so I just went with multiplying the maximum and dividing the minimum by a factor of 2
    for Axis_inc in range(2):
        Min_list.append(min([X_axis, Y_axis][Axis_inc]) / Graph_window_margin)
        Max_list.append(max([X_axis, Y_axis][Axis_inc]) * Graph_window_margin)
    
    cmap = cm.jet # Colormap for the 3rd axis
    cmap = colors.LinearSegmentedColormap.from_list('jet_truncated', cmap(np.linspace(0.2, 0.95, 100)))
    
    Z_axis_cleaned = list(filter(lambda Item: Item != None, Z_axis)) # Positivity rate to color converter
    norm = colors.Normalize(vmin = 0, vmax = max(Z_axis_cleaned), clip = True)
    mapper = cm.ScalarMappable(norm = norm, cmap = cmap)
    
    plt.close() # Initialise plotting area. A simple "plt.clf()" doesn't work to erase everything and prompts glitches after the 2nd execution of the code, forcing us to close the figure and reopen it
    fig = plt.figure("Scatter graph of COVID data")
    fig.set_size_inches(tuple(1/fig.dpi * np.array([1920, 1080])))
    ax = fig.gca()
    
    manager = plt.get_current_fig_manager() # Adapt the matplotlib window to the screen
    manager.window.showMaximized()
        
    Data_frames = zip(COVID_data_scatter.keys(), COVID_data_scatter.values()) # Transform the first level of dictionnary into a list because we need to have access to the keys of that first level during the creation of the animation frames
    Animation_frames = [] # List where all the matplotlib objects for the animation will be stored
    for Frame in Data_frames:
        Date = Frame[0]
        
        Points_to_display, Positivity_rate_list, Points_colors = [], [], []
        
        Countries_displayed = list(Frame[1].keys())
        
        for Country in Countries_displayed: # For each country...
            Country_coords = Frame[1][Country][:2]
            Positivity_rate = Frame[1][Country][2]
            
            Points_to_display.append(Country_coords)
            
            if Positivity_rate != None: # If there is a positivity rate for that country, it is plotted with the color it corresponds to on the colormap
                Positivity_rate_list.append(Positivity_rate)
                Points_colors.append(mapper.to_rgba(Positivity_rate))
            else: # Otherwise, it appears in #ABB7B7 gray and a "-1" is appended to the list of positivity rates. That way, these points will be in last after the sorting in descending order in a few lines
                Positivity_rate_list.append(-1)
                Points_colors.append((0.6627, 0.6627, 0.6627, 1))
        
        All_points_info = list(zip(Countries_displayed, Points_to_display, Positivity_rate_list, Points_colors)) # Group everything, sort the points based on the positivity rate and then seperate everything to get the same objects as before but sorted
        All_points_info.sort(key = lambda x: x[2])
        All_points_info = list(zip(*All_points_info))
        
        Countries_displayed = list(All_points_info[0])
        Points_to_display = list(All_points_info[1])
        Positivity_rate_list = list(All_points_info[2])
        Points_colors = list(All_points_info[3])
        
        X_list_frame, Y_list_frame = zip(*Points_to_display) # Separate X and Y axes and plot the points
        scatter = ax.scatter(X_list_frame, Y_list_frame, c = Points_colors, s = np.pi * (Marker_ray*72/fig.dpi)**2, linewidth = 0.5, edgecolors = 'black') # Marker ray is the radius of the circle in pixels but s is the area of the circle in points. We have to convert the pixels in points (1 inch = 72 points = Screen dpi) then apply area = pi * radius²
        
        # Note: ax.scatter plots the points one by one so the last elements of the lists will be above the firsts. Since the X and Y axes are sorted in ascending order of positivity rate, the last points (high positivity rates) will be on top. This is on purpose because these are the most interesting ones
        
        Text_date = ax.text(0.02, 0.97, Date, transform = ax.transAxes, fontsize = Date_fontsize, verticalalignment = 'top', horizontalalignment = 'left', bbox = dict(boxstyle = 'round', facecolor = 'white', alpha = 0.9, pad = 0.3)) # Display the date
        
        fig.tight_layout() # Annotations_frame() requires the use of lines regarding the size of the plotting area. For them to work properly, we have to virtually draw the elements, which is why we use fig.tight_layout() in the middle of the creation of the animation frames
        
        Countries_to_annotate, Annotations_mask = Annotations_frame(Points_to_display[::-1], Countries_displayed[::-1], (Min_list[0], Max_list[0], Min_list[1], Max_list[1])) # Decide which countries to annotate and which not to
        
        Annotation_list = []
        for Country, Country_coords in zip(Countries_to_annotate.keys(), Countries_to_annotate.values()): # Annotate countries
            Annotation_list.append(ax.annotate(Country, Country_coords, textcoords = 'offset pixels', xytext=(0, Marker_ray + Annotation_fontsize/72*fig.dpi*0.5 + Annotation_offset), ha='center', va='center', fontsize = Annotation_fontsize))
        
        if Display_annotations_mask: # If something goes wrong during an edit, the user can still display the annotations outline
            ax_tw_x = ax.twinx() # Duplicate axis. Compulsory because the graph is in logscale and an image cannot be properly displayed in logscale
            ax2 = ax_tw_x.twiny()
            
            mapper_mask = cm.ScalarMappable(norm = colors.Normalize(vmin = 0, vmax = 1, clip = True), cmap = cm.gray) # Convert array of bools into array of colors then display the image
            Annotations_mask_im = mapper_mask.to_rgba(np.rot90(np.invert(Annotations_mask) + np.zeros(Annotations_mask.shape)), alpha = 0.3)
            Annotations_mask_ax = ax2.imshow(Annotations_mask_im, extent = [Min_list[0], Max_list[0], Min_list[1], Max_list[1]], aspect = 'auto')
            
            ax_tw_x.axis('off') # Not display axes of the image
            ax2.axis('off')
            
            Animation_frames.append([scatter, Text_date, Annotations_mask_ax] + Annotation_list)
        
        else: Animation_frames.append([scatter, Text_date] + Annotation_list)
        
    ax.set_title("COVID-19 pandemic - %s vs. %s" % tuple(COVID_data_scatter_names[:2][::-1]), fontsize = Title_fontsize, pad = Title_pad)
    
    ax.set_xlabel(COVID_data_scatter_names[0], fontsize = Axis_label_fontsize)
    ax.set_ylabel(COVID_data_scatter_names[1], fontsize = Axis_label_fontsize)
    
    ax.set_xlim(Min_list[0], Max_list[0])
    ax.set_ylim(Min_list[1], Max_list[1])
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    ax.grid(linestyle = '--', linewidth = 1.5, which = 'major')
    ax.grid(linestyle = '--', linewidth = 0.5, which = 'minor')
    
    ax.set_axisbelow(True)
    
    ax.tick_params(axis='x', labelsize = Axis_tick_fontsize)
    ax.tick_params(axis='y', labelsize = Axis_tick_fontsize)
    
    Formatter_list = [] # Display axes graduations as multiples of 10 (rather than 10^n) and find how many decimal places to display
    for Axis_inc in range(2):
        if Min_list[Axis_inc] < 1: Min_axis_log = int(np.abs(np.floor(np.log10(Min_list[Axis_inc])))) - 1
        else: Min_axis_log = 0
        
        Formatter_list.append('%.' + str(Min_axis_log) + 'f')
    
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter(Formatter_list[0]))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter(Formatter_list[1]))
    
    Divider = make_axes_locatable(ax) # Display 3rd axis (colors). Using make_axes_locatable() allows for better tight_layout results
    cax = Divider.append_axes('right', size = '2%', pad = 0.3)
    cbar = fig.colorbar(mapper, cax = cax)

    cbar.ax.set_ylabel(COVID_data_scatter_names[2], fontsize = Axis_label_fontsize, labelpad=Axis_label_pad)
    cbar.ax.tick_params(axis='y', labelsize = Axis_tick_fontsize)
    cbar.ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax = 1, decimals = 0)) # Set axis graduations as percentage with no decimal places
    
    ani = animation.ArtistAnimation(fig, Animation_frames, blit = True, interval = Animation_interval)
    
    fig.tight_layout()
    fig.show()
    
    return ani, COVID_data_scatter_names