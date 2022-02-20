# WdSI_projekt
Zasada działania projektu to rozpoznawanie znaków drogowych i rozróżnienie ich na dwie kategorie, "speedlimit" oraz "other".
Zdjęcia oznaczone opisem "speedlimit" powinny zawierać znak ograniczenia prędkości, natomiast jeżeli na obrazie nie znajduje się taki znak, to program przypisuje etykietę "other".
Po uruchomieniu programu zaczyna on uczenie na podstawie danych z folderu "train" (funkcja learn_bovw()).
Następnie obliczane są wektory cech dla każdego z obrazów (funkcja extract_features()).
Po obliczeniu wektorw cech funkcja train() uczy model RFC (RandomForestClassifier).
Teraz można przystąpić do obliczenia wektorów cech oraz do predykcji danych z folderu test na podstawie wyuczonego modelu.
Wpisanie "classify" na standardowe wejście programu, na podstawie wprowadzonych informacji zwraca etykietę jaki znak występuje na podanych zdjęciu.
Wpisanie "detect" na standardowe wejście programu, powoduje wypisanie poszczególnych informacji (nazwa, liczba wykrytych obiektów, współrzędne zaznaczonego obszaru) o zdjęciach jakie znajdują się w folderze test.
