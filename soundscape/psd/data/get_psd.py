from urllib.request import urlopen
import pandas as pd

seismo_data = pd.read_csv('/home/irseppi/REPOSITORIES/parkshwynodal_supp/input/full_nodes.txt', sep="|")
stations = seismo_data['Station']
channels = ['1', '2', 'Z']
for channel in channels:
    for station in stations:
        station = str(station)
        xml_file = "psd_stations/psd_"+station+"_DP"+channel+".xml"
        xml = open(xml_file, "w")
        try:
            xml_url = f'https://service.iris.edu/mustang/noise-psd/1/query?net=ZE&sta={station}&loc=--&cha=DP{channel}&quality=D&starttime=2019-02-11T00:00:00&endtime=2019-03-26T00:00:00&correct=true&format=xml&nodata=404'
            xml.write(urlopen(xml_url).read().decode('utf-8'))
        except Exception as e:
            print(f"Error fetching {station}: {e}")
        finally:
            xml.close()
