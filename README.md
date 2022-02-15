# WdSI_projekt
Zasada działania projektu to rozpoznawanie znaków drogowych i rozróżnienie ich na dwie kategorie, "speedlimit" oraz "other".
Zdjęcia oznaczone opisem "speedlimit" powinny zawierać znak ograniczenia prędkości, natomiast jeżeli na obrazie nie znajduje się taki znak,
to program przypisuje etykietę "other".
Po uruchomieniu programu zaczyna on uczenie na podstawie danych z folderu "train" (funkcja learn_bovw()).
Następnie obliczane są deskryptory dla każdego z obrazów (funkcja extract_features()).
Po obliczeniu deskryptorów funkcja train() uczy model RFC (RandomForestClassifier).
Teraz można przystąpić do obliczeniu deskryptorów oraz do predykcji (funkcja extract_features() oraz predict()) danych z folderu test na podstawie wyuczonego modelu.
Po wymienionych operacjach program wyświetla ilość obrazów poprawnie dopasowanych oraz dopasowanych błędnie.
Następnie program oczekuje na wpisanie komendy "classify" lub "detect" (funkcja input_data()).
Wpisanie "classify", na podstawie wprowadzonej nazwy zdjęcia zwraca etykietę jaki znak występuje na podanych zdjęciu.
Wpisanie "detect" powoduje wypisanie poszczególnych informacji (nazwa, liczba wykrytych obiektów, współrzędne zaznaczonego obszaru) o zdjęciach jakie znajdują się w folderze test.
Program zawiera funkcje, które nie zostały wykorzystane z powodu braku predykcji współrzędnych obszaru obiektów 
(obliczanie współczynnika IoU. (funkcje create_rectangles() oraz bb_intersection_over_union())
