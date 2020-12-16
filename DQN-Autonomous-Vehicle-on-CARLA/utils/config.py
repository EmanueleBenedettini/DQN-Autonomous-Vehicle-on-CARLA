import json 
import os

CONFIG_DIR = "./config/"

def generate_default_config_data():
    data = []

    #Generate default config
    connection = {'server_ip': '127.0.0.1', 'server_port': 2000}

    #concat config in one file
    data = {'connection': connection}

    return data

def load_data(filename):
    try:
	    with open(CONFIG_DIR + filename) as json_data_file:
		    data = json.load(json_data_file)
    except FileNotFoundError:
        data = generate_default_config_data()
        save_data(filename, data)
    finally:
	    return data
	
	
def save_data(filename, data):
	try:    
		os.makedirs(CONFIG_DIR)
	except FileExistsError:
		# directory already exists
		pass

	with open(CONFIG_DIR + filename, "w") as outfile:
		json.dump(data, outfile)
	return

