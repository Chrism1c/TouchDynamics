devices.csv

	device_id, density, resolution, os_version, xdpi, ydpi


users.csv


	user_id, gender, birthyear, touch_experience_level

	gender: 0 - male, 1 - female
	touch_experience_level: 0, 1, 2, 3 (0 - no experience, 3 - highly experienced user)


rawdata.csv
	device_id, user_id, doc_type, timestamp, action, phone_orientation, x_coordinate, y_coordinate, pressure, finger_area

	doc_type: 1, 2 (1: imagge_gallery activity , 2: reading activity)
	action: 0, 1, 2 (0: action_down, 1: action_up, 2: action_move)


