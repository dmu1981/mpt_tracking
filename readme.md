# Machine Perception and Tracking 
Dies ist die zweite Aufgaben für das MPT Modul im Studiengang DAISY.
In diesem Modul müssen sie mehrere Filter implementieren um Meßreihen zu verarbeiten und eine möglichst optimale Zustandsschätzung zu erreichen. 

Dabei gibt es verschiedene dynamische Modelle und zu jedem Modell
eine Vielzahl von Meßreihen. Jede Meßreihe wiederum besteht aus mehreren Messungen zu verschiedenen Zeitpunkten. Ihre Aufgabe ist es zu jeder Messung eine möglichst optimale Schätzung für den zugrundeliegenden Zustand zu berechnen und diese dann zurückzugeben. 
Dazu implementieren Sie in einer eigenen Datei s.g. Filter in dieser Form:

    class DummyFilter():
      def __init__(self, shape):
        self.shape = shape

      def reset(self, measurement):    
        return np.zeros(self.shape)
    
      def update(self, dt, measurement):  
        return np.zeros(self.shape)

Dieser mitgelieferte Dummy-Filter tut gar nichts, er gibt zu jedem Zeitpunkt einfach ein 0-Array zurück. Die "reset" Methode wird
zu Beginn jeder Meßreihe aufgerufen mit dem ersten Meßwert. Anschließend wird die "update" Methode für jeden weiteren Meßwert 
aufgerufen. DT beschreibt die Zeit die seit der letzten Messung vergangen ist. Beide Methoden müssen ein numpy Array zurückgeben
welches den zu schätzenden Zielzustand beschreibt. 

## Verwendung des Programms
In der Datei config.py können Sie ihre Filter hinzufügen. 
Dabei ist die Variable "filters" ein Dictionary, dessen Keys
dem jeweiligen Teamnamen entsprechen. Zu ihrem Team instantieren Sie 
dann für jedes der Probleme ihren jeweiligen Filter. 

    # TODO: Add your filters here
    filters = {
      "Dummy": {
        "color": "#008000", # green
        "constantposition": dummy.DummyFilter(2)
      }
    }

Rufen Sie danach die main.py Methode auf und übergeben Sie als Parameter den Schalter "--mode". Als Parameter für diesen Schalter geben Sie das zu testende Problem an, z.B. "constantposition". Geben 

    python main.py --mode=constantposition

Die Ausgabe enthält den RMSE (Root-Mean-Square-Error) aller registrierten Filter für das gegebene Problem. Niederiger ist besser.

In dieser Konfiguration wird immer nur die allererste Meßreihe gefiltert. Das Ergbeniss ist also deterministisch und sie können erste Optimierungen ihres Filters vornehmen. Mit dem optionalen Schalter --index können Sie eine andere Meßreihe ausprobieren, zu jedem Problem sind mehrere verschiedene Meßreihen hinterlegt. 

    python main.py --mode=constantposition --index=5

Der Schalter --debug 

    python main.py --mode=constantposition --debug

aktiviert eine Visualisierung der Meßreihe inklusive der wahren (von Ihnen zu schätzenden) Zustände, den Messungen sowie ihren jeweiligen Schätzungen. Auf der rechten Seite sehen Sie ihren jeweiligen Meßfehler als Zeitreihe sowie den RMSE (gestrichelte Linie).

Mit dem Schalter --all 

    python main.py --mode=constantposition --all

können Sie ihren Filter über alle Meßreihen des jeweiligen Problems laufen lassen. Sie sehen dann den mittleren RMSE über alle Meßreihen. Die Visualisierung funktioniert in diesem Modus allerdings nicht!

Es gibt einen speziellen Modus mit Namen "all", der entsprechende Aufruf ist

    python main.py --mode=all

In diesem Modus werden alle Probleme nacheinander durchgerechnet und eine Rang-basierte Bewertung aller Teams durchgeführt. Rang-basiert bedeutet das zunächst innerhalb eines Problems die Filter der jeweiligen Teams anhand des RMSE (siehe oben) bewertet werden. Die Teams werden anhand dieses Scores sortiert wodurch ein Rang erzeugt wird (bestes Team, zweitbestes Team, etc...). Teams auf dem ersten Platz erhalten 1 Punkt, auf dem zweiten Platz gibt es zwei Punkte, und so weiter. Über die verschiedene Probleme hinweg werden dann diese Punkte addiert und wieder sortiert. Durch diesen Mechanismus wird die Gesamtbewertung nicht durch die tatsächliche Höhe der RMSE Werte beeinflusst, d.h. eine einzelne, sehr schlechte RMSE-Bewertung kann nicht die Gesamtbewertung kaputt machen.

## Problem 1 - Statisches Objekt (Direkte Messung)
Dieses Problem heißt in der config.py Datei **constantposition**!

Ein Objekt befindet sich an einem unbekannten Ort. Es sind die beiden Koordinaten des Objektes (2D) zu schätzen. Das Objekt ist statisch, bewegt sich also nicht,d.h. es gilt stets

    x(t) = x(t-1)

Die Messung des Ortes erfolgt direkt, d.h. der übergebene Meßvektor gibt direkt die Koordinaten des Objektes an. Jede Messung ist durch unabhängiges, normalverteiltes Rauschen mit einer Standardabweichung von 0.2 Einheiten pro Achse überlagert, d.h.

    z(t) = x(t) + et

wobei et ~ N(0, 0.04) das normalverteilte Rauschen darstellt.    

Der Meßvektor ist also 2-dimensional und beschreibt die verrauschte Position. Der zu schätzende Zustand ist die ebenfalls 2-dimensionale wahre Position des Objektes.

## Beispiel für das erste Problem
Um einen ersten Eindruck davon zu bekommen was sie
tun müssen können Sie selbst den folgenden "NoFilter" ausprobieren.
Dieser tut gar nichts sondern gibt einfach nur die ersten beiden Meßwerte direkt wieder zurück. Dies ist eine (wenn auch sehr schlechte) Möglichkeit das Problem zu lösen und sie erhalten einen hohen RMSE (Root-Mean-Squared-Error) mit diesem Filter.

Erzeugen Sie dazu eine neue Datei und kopieren Sie diese Klasse hinein.

    class NoFilter():
      def __init__(self):
        pass

      def reset(self, measurement):    
        return measurement[:2]
      
      def update(self, dt, measurement):  
        return measurement[:2]

Importieren Sie ihre Datei nun in der config.py und fügen Sie
einen entsprechenden Eintrag im filters-Dictionary hinzu. Der
entsprechende Eintrag könnte also z.B. so aussehen:

    # TODO: Add your filters here
    filters = {
      "Dummy": {
        "color": [0.2, 0.2, 0.4],
        "constantposition": dummy.DummyFilter(2),
      },
      "NoFilter": {
        "color": [0.5, 0.2, 1.0],
        "constantposition": NoFilter(),
      },  
    }

Wenn Sie nun das Programm auf dem "constantposition"-Problem starten erhalten Sie folgende Ausgabe

    python main.py --mode=constantposition --all             
       NoFilter  :     0.1985 (Best run was index 12, Worst run was index 21)
       Dummy     :     0.3592 (Best run was index 15, Worst run was index 7)

Die Anzeige ist sortiert, d.h. der beste Filter steht immer oben. In diesem Fall ist der "NoFilter" also bereits besser als der "Dummy"-Filter. Dies ist verständlich insoweit als das der Dummy-Filter ja immer die Position (0,0) schätzt (unabhängig von der Messung) während der "NoFilter" wenigstens noch den letzten Meßwert zurückgibt. Sie können sich den besten Durchlauf für den NoFilter konkret anschauen indem Sie das Programm so starten

    python main.py --mode=constantposition --index=12 --debug
       NoFilter  :     0.1769
       Dummy     :     0.3863

Der RMSE-Fehler des NoFilter in diesem Durchlauf war mit 0.1769 also minimal.

Schauen Sie sich auch den schlechtesten Durchlauf ihres Filter an, also

    python main.py --mode=constantposition --index=21 --debug  
       Dummy     :      0.143
       NoFilter  :     0.2207

Diesmal ist sogar der Dummy-Filter deutlich besser obwohl der "NoFilter" offensicht die bessere Strategie verfogt. Dies deutet darauf hin das der wahre Zustand ohnehin schon nah am Punkt (0,0) gelegen hat.

Merke: Es nicht möglich ist die Filter anhand einzelner Meßreihen zu vergleichen. Vergleichen Sie immer auf allen mitgelieferten Meßreihen indem sie die --all Option verwenden. 

Schauen wir uns noch den schlechsten Run des Dummy Filters (7 in diesem Fall) an, so finden wir

    python main.py --mode=constantposition --index=15 --debug  
       NoFilter  :     0.1905
       Dummy     :     0.7023

In diesem Szenario muß der wahre (zu schätzende Zustand) also weit weg von (0,0) gelegen haben, weswegen der Dummy-Filter hohe Fehler erzeugt. Der NoFilter schneidet besser ab. Tatsächlich ist es kein Zufall das der RMSE des NoFilter im Mittel (über alle Durchläufe) bei etwa 0.2 liegt... Dies entspricht ja genau
dem Meßrauschen laut Aufgabenstellung (vergleichen Sie den entsprechenden Aufgabentext).

Überlegen Sie nun wie sie die Schätzung verbessern können. Ihr Ziel ist es einen möglichst niedrigen RMSE über alle Meßreihen zu erhalten. Hinweis: Mit meiner Implementierung eines sinnvollen Filters erreiche ich bei diesem Problem einen RMSE von 0.0368

    python main.py --mode=constantposition --all
        Mueller   :     0.0368 (Best run was index 33, Worst run was index 18)
        NoFilter  :     0.1985 (Best run was index 12, Worst run was index 21)
        Dummy     :     0.3592 (Best run was index 15, Worst run was index 7)
       
## Problem 2 - Statisches Objekt mit unterschiedlichem Meßrauschen 
Dieses Problem heißt in der config.py Datei **randomnoise**!

Wie bei Problem 1 befindet sich ein statisches Objekt an einem unbekannten Ort, es ist also wieder

    x(t) = x(t-1)


Wieder sollen die Koordinaten (2D) geschätzt werden. Die Messung ist ebenfalls
wieder direkt (in kartesischen Koordinaten) und von Rauschen überlagert. Diesmal
ist jedoch jede Messung mit einem individuellen Rauschen überlagert dessen jeweilige Kovarianz gegeben ist. Es ist diesmal also

    z(t) = x(t) + et

mit et ~ N(0, Rt), wobei Rt die (gegebene) Kovarianz des Meßrauschens in diesem Zeitschritt darstelt. 

Der Meßvektor ist 6-dimensional. Die ersten beiden Dimensionen
entsprechen den kartesischen Koordinaten des gemessenen (verrauschten) Zustands. Die letzten vier Dimensionen enthalten die 4 Einträge der 2x2 Kovarianz-Matrix für das jeweilige Meßrauschen. Sie können mit 

    Rt = measurement[2:].reshape(2,2)

diese Matrix rekonstruieren. Verwenden Sie das bekannte Meßrauschen und dessen Korrelation um ihre Schätzung zu verbessern!

## Problem 3 - Statisches Objekt (Winkelmessung)
Dieses Problem heißt in der config.py Datei **angular**!

Wieder befindet sich ein Objekt an einem unbekannten Ort. Es sind die beiden Koordinaten des Objektes (2D) zu schätzen. Das Objekt ist statisch, bewegt sich also nicht, d.h. es ist wieder

    x(t) = x(t-1)

Die Messung des Ortes erfolgt direkt, es wird jedoch der Abstand zum Ursprung sowie der Winkel zum Objekt gegenüber der x-Achse angegeben ([Polar-Koordianten](https://de.wikipedia.org/wiki/Polarkoordinaten)).

Dabei ist die Abstandsmessung (r) in der ersten Dimension und die Winkelmessung (phi) in der zweiten Dimension kodiert. Die Abstandsmessung ist mit einem normalverteilten Rauschen mit Standardabweichung 0.1 überlagert. Die Winkelmessung erfolgt in Radians (Bogenmaß) und ist mit ebenfalls normalverteiltem Rauschen mit Standardabweichung 0.05 (etwa 3°) überlagert. Es ist also

    z(t) = (r, phi) + et

mit r und phi wie unten und et ~ N(0, R) mit 
        
        | 0.0100  0.0000 |
    R = |                |
        | 0.0000  0.0025 |

Der Meßvektor ist also 2-dimensional und beschreibt die verrauschte Position in Polarkoodinaten. Der zu schätzende Zustand ist wieder die ebenfalls 2-dimensionale wahre Position des Objektes, jedoch in kartesischen Koordinaten (also X und Y).

Hinweis: Sie können die [arctan2](https://de.wikipedia.org/wiki/Arctan2) Methode verwenden um kartesische Koordinaten in Polarkoordinaten umzurechnen

    r   = sqrt(x²+y²)
    phi = arctan2(y,x) # Achtung: Zuerst Y, dann X!

Die Rückrechnung erfolgt trivial über

    x = r * cos(phi)
    y = r * sin(phi)

## Problem 4 - Konstante Geschwindigkeit (Einzelne Messung)
Dieses Problem heißt in der config.py Datei **constantvelocity**!

Ein Objekt befindet sich an einem unbekannten Ort und bewegt sich mit unbekannter 
aber konstanter Geschwindigkeit fort. Das Objekt bewegt sich nach den Gesetzen der Newtonschen Mechanik, d.h. es ist

    x(t) = x(t-1) + dt * v
    
Gemessen werden kann wie in Problem 1 nur die Position mit unkorreliertem Meßrauschen. Wie bei Problem 1 wird die Messung durch ein unabhängiges, normalverteiles Meßrauchen in X und Y Richtung mit einer Standardabweichung von 0.2 überlagert, d.h. es ist wieder

    z(t) = x(t) + et

mit et ~ N(0, 0.04)

Der zu schätzende Zustand ist die 2-dimensionale Position 
des Objektes. Beachten Sie das die DT-Werte aus der Simulation nicht konstant sind!

## Problem 5 - Konstante Geschwindigkeit (Mehrere Messung)
Dieses Problem heißt in der config.py Datei **constantvelocity2**!

Wie bei Problem 4 befindet sich ein Objekt an einem unbekannten Ort und bewegt sich mit unbekannter aber konstanter Geschwindigkeit fort, d.h. es ist wieder

    x(t) = x(t-1) + dt * v

Gemessen werden kann wie in Problem 1 nur die Position mit unkorreliertem Meßrauschen, allerdings erhalten Sie diesmal mehrere (konkret: fünf!) unabhängige Messungen. Jede Messung ist wieder mit unabhängigem Meßrauschen überlagert. Die Standardabweichung für jede Messung ist in jeder Phase zufällig und wird ebenfalls als Teil des Meßvektors übergeben. Der Meßvektor z ist also 20-Dimensionnal, wobei die ersten 10 Dimensionen den fünf Messungen (jeweils eine X- und eine Y-Koordinate) entsprechen. Die letzte 10 Dimensionen geben die jeweilige Standardabweichung des Meßrauschens an.

Es ist also 

    z(t) = [ z1(t), z2(t), z3(t), z4(t), z5(t), 
             sigma_1, sigma_2, sigma_3, sigma_4, sigma_5, sigma_6, sigma_7, sigma_8, sigma_9, sigma_10 ]

mit zi(t) = x(t) + e_it und e_it ~ N(0, sigma_i)

Beispiel: Ist z.B. der Meßvektor z gegeben als

    z(t) = [2.46 2.41 2.02 2.86 2.43 2.57 2.59 2.63 2.55 2.52 
            0.25 0.42 0.38 0.67 0.07 0.98 0.32 0.13 0.44 0.30]

so bedeutet dies das wir 5 Positionsmessungen mit jeweiligen Rauschtermen

    z1(t) = [2.46 2.41] + N(0,diag(0.25, 0.42))
    z2(t) = [2.02 2.86] + N(0,diag(0.38, 0.67))
    z3(t) = [2.43 2.57] + N(0,diag(0.07, 0.98))
    z4(t) = [2.59 2.63] + N(0,diag(0.32, 0.13))
    z5(t) = [2.55 2.52] + N(0,diag(0.44, 0.30))

haben. Dabei ist diag(x,y) eine Diagonalmatrix mit x und y als Elemente auf der Hauptachse.

Der zu schätzende Zustand ist die 2-dimensionale Position des Objektes. Beachten Sie wieder das die DT-Werte aus der Simulation nicht konstant sind!

Hinweis: Sie können die Positionsmessungen und Messunsicherheiten so aus dem gegebenen Messvektor rekonstruieren:

    z = measurement[:10]
    R = np.diag(measurement[10:])

## Problem 6 - Constant Turn Rate
Dieses Problem heißt in der config.py Datei **constantturn**!

Wie bei Problem 5 befindet sich ein Objekt an einem unbekannten Ort und bewegt sich mit einer Anfangsgeschwindigkeit fort, d.h. es ist wieder

    x(t) = x(t-1) + dt * v(t-1)

Allerdings dreht der Geschwindigkeitsvektor des Objektes mit einer konstanten "Turn Rate" a (Winkelgeschwindigkeit), d.h. es ist

           | cos(a * dt)   -sin(a * dt) |    
    v(t) = |                            | * v(t-1)  
           | sin(a * dt)    cos(a * dt) |   

Gemessen werden kann wie in Problem 1 nur die Position mit unkorreliertem Meßrauschen, allerdings erhalten Sie genau wie in Problem 5 mehrere (konkret: fünf!) unabhängige Messungen. Jede Messung ist wieder mit unabhängigem Meßrauschen überlagert. Die Standardabweichung für jede Messung ist in jeder Phase zufällig und wird ebenfalls als Teil des Meßvektors übergeben. Der Meßvektor z ist also wieder 20-Dimensional, wobei die ersten 10 Dimensionen den fünf Messungen (jeweils eine X- und eine Y-Koordinate) entsprechen. Die letzten 10 Dimensionen geben wieder die jeweilige Standardabweichung des Meßrauschens an. Vergleichen Sie dazu das Problem 5. 

## Modus Operandi und Bewertung
Bewertet wird vor allem ob sie das Problem mathematisch korrekt
modellieren, also insbesondere ob Sie die Methoden aus der Vorlesung korrekt umsetzen. Während ihrer Vorstellung müssen sie das konkret verwendete mathematische Modell vorstellen und diskutieren. Für eine korrekte Modellierung erhalten Sie die volle Punktzahl, ist ihre Modellierung in Ansätzen zwar richtig aber im Detail fehlerhaft erhalten Sie nur die Hälfte der Punkte. Eine komplett fehlerhafte Modellierung resultiert natürlich in 0 Punkten für das Teilproblem. 

Da die Probleme unterschiedlich schwierig zu modellieren sind gibt es auch unterschiedlich viele Punkte für jedes Problem. Insgesamt können Sie in dieser Aufgabe 200 Punkte verdienen. Für die ersten 3 Probleme erhalten Sie maximal jeweils 20 Punkte (60 Punkte gesamt), für die letzten 3 Probleme erhalten Sie maximal jeweils 40 Punkte (120 Punkte). 

Die letzten 20 Punkte kann nur ein Team erlangen, und zwar das Team welches über alle Probleme den besten Rang erzielt, also beim Aufruf

    python main.py --mode=all      
    overall
      Mueller   :        0.0
      NoFilter  :        8.0
      Dummy     :       10.0

ganz oben genannt wird. Dazu müssen Sie natürlich zunächsteinmal für alle Probleme jeweils
einen Filter implementieren. 

