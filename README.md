# WdSI projekt

# Działanie projektu
Działanie projektu to rozpoznawanie znaków drogowych z podziałem na dwie kategorie - `speedlimit` oraz `other`. 
Program początkowo przeprowadza uczenie na podstawie danych z folderu `train` z wykorzystaniem procedury ekstrakcji cech w algorytmie BoVW.
Po wprowadzeniu na standardowe wejście programu komendy `classify` oraz po podaniu odpowiednich danych takich jak ilość plików do przetworzenia, nazwa zdjęcia,
liczba wycinków na danym zdjęciu i wspołrzędnych prostokąta zawierającego wycinek obrazu, program zwraca etykiete `speedlimit` lub `other`.
Po wprowadzeniu na standardowe wejście programu komendy `detect`, program zwraca nazwe znaku, ilość wykrytych na nim znaków oraz wspołrzędne prostokąta zawierającego wycinek obrazu.

# Biblioteki
- import os
- import numpy as np
- import cv2
- import xml.etree.ElementTree as ET
- from sklearn.ensemble imoprt RandomForestClassifier
- from os import listdir
- from os.path import isfile, join

# Funkcje w projekcie
- `learn_bovw(data, t_p)` - odpowiedzialna za stowrzenie słownika punktów kluczowych na zdjęciach treningowych.
- `extract_features(data, path, cropped_box_)` - odpowiedzialna za ekstrakcje wektora cech ze zdjęć treningowych oraz testowych.
- `build_data(_all_data_, all_files, tree)` - odpowiedzialna za odpowiednie dostosowanie danych wejściowych, aby działanie programy było możliwe.
- `train(data)`, `train_object_number(data)` oraz `train_box(data)` - odpowiedzialne odpowiednio za uczenie algorytmu rozpoznawania rodzaju znaku, ilości znaków na zdjęciu oraz wzpółrzędnych znaku lub znaków.
- `predict(rf, data)`, `predict_object_number(rfobj, data)` oraz `predict_box(rfbox, data)` - odpowiedzialne odpowiednio za predykowanie rodzaju znaku, ilości znaków na zdjęciu oraz wzpółrzędnych znaku lub znaków.
- `input_data(data, test_path, rf, rf_obj, rf_box)` - odpowiedzialna za prowadzenie danych do klasyfikacji lub detekcji
- `set_data()` - odpowiedzialna za odczyt danych z odpowiednich folderów.
