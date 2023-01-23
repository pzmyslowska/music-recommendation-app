from calendar import c
from tkinter import *

root=Tk()
root.title('Music Recommendation App') # nazwa programu (wyświetlanie na oknie programu)
root.iconbitmap('C:/Users/budaa/Documents/VSCode/Music_recmmendation_system/play.ico') # ustawienie ikony okna programu

welcome = Label(root, text="Podaj nazwę utworu oraz jego wykonawcę, a następnie wciśnij przycisk \"Dodaj\"")
welcome.grid(row=0, column=0, columnspan=3)

songs = []
# zwraca wpis 
def display_data():
    s_name = s_name_entry.get()
    s_name_label.config(text= s_name)

    s_artist = s_artist_entry.get()
    s_artist_label.config(text= s_artist)

    songs.append([s_name, s_artist]) # dodawanie utworów do listy

    print(songs)
    print(songs[0][0])

    display = Label(root, text="**** DODANE UTWORY ****", font = 9) 
    display.grid(row=5, column=0)

    display = Label(root, text="UTWÓR") 
    display.grid(row=6, column=0)

    display = Label(root, text="WYKONAWCA") 
    display.grid(row=6, column=1)

    # wyświetlanie wpisanych utworów
    for i in range(len(songs)):
        for j in range(2):
            song = Label(root, text=songs[i][j])
            song.grid(row=i+7, column=j)

def get_songs():
    # opisuje pole - input utworu
    s_name_des = Label(root, text="Utwór: ")
    s_name_des.grid(row=1, column=0)

    #opisuje pole - input wykonawcy
    s_artist_des = Label(root, text="Wykonawca: ")
    s_artist_des.grid(row=2, column=0)

    # pole - input utworu
    global s_name_entry
    s_name_entry = Entry(root, width=40)
    s_name_entry.grid(row=1, column=1)

    #pole - input wykonawcy
    global s_artist_entry
    s_artist_entry = Entry(root, width=40)
    s_artist_entry.grid(row=2, column=1)

    # # output utworu
    global s_name_label
    s_name_label = Label(root, text="")

    # # output wykonawcy
    global s_artist_label
    s_artist_label = Label(root, text="")
    
get_songs()

# przycisk do wyświetlenia wpisu
click_button = Button(root, text="Dodaj", command=display_data)
click_button.grid(row=4, column=1)


root.mainloop()