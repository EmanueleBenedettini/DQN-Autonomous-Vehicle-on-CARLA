import json 

def generate_default_config_data():
    data = []

    connection = {'server_ip': '127.0. 0.1', 'server_port': 2000}

    data = {'connection': connection}

    return data

def load_data(filename):
    try:
	    with open(filename) as json_data_file:
		    data = json.load(json_data_file)
    except FileNotFoundError:
        data = generate_default_config_data()
        save_data(data)
    finally:
	    return data
	
	
def save_data(filename, data):
	with open(filename, "w") as outfile:
		json.dump(data, outfile)
	return

