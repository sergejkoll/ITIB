import data

if __name__ == '__main__':
    districts = data.init_district_centers(data.district_centers)

    j_file = data.read_json('data', 'r')
    hospitals = data.init_hospitals(j_file)

    with open('results.txt', 'w') as file:
        data.analysis(hospitals, districts).get_statistic(file)
