import DataService
import StatisticsService
import ShowStructurer

dm = DataService.DataManager()

def lesgo(name, file):
    dm.extract_data(name, file)
    segments = StatisticsService.segment(name, dm.get_song(name), ["drums", "other"])
    print("------------------SEGMENTS--------------------")
    for segment in segments:
        print(segment)

name = "thunder2"
file_path = f"./songs/{name}.mp3"
lesgo(name, file_path)

structurer = ShowStructurer.ShowStructurer(dm)
structurer.generate_show(name)


# with open(write_path, 'a') as file:
#     # Write a new line
#     file.write('\nThis is a new line')
# print("done")q