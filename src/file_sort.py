import os
import re
import sys

sys.path.append(os.path.join(os.path.join(os.path.dirname(sys.path[0]))))

home_dir = os.path.join(os.path.join(os.path.dirname(sys.path[0])))

def get_latest_weights_file():
	home_dir = os.path.join(os.path.join(os.path.dirname(sys.path[0])))+'/data/logs'

	files = [file for file in os.listdir(os.path.join(home_dir)) if (file.lower().endswith('.h5'))]
	files.sort(reverse = True)

	return os.path.abspath(files[0])


