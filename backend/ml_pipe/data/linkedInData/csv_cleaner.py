import pandas as pd
import os
import sys

def clean_csv(input_file):
    """
    Vereinfacht die CSV-Datei auf die wesentlichen Spalten.
    """
    try:
        # Pfad zur CSV-Datei
        csv_folder = "backend/ml_pipe/data/datafiles/"
        input_path = os.path.join(csv_folder, input_file)
        
        if not os.path.exists(input_path):
            print(f"Fehler: Datei nicht gefunden: {input_path}")
            return None
            
        # Lese die CSV-Datei
        print(f"Lese Datei: {input_file}")
        df = pd.read_csv(input_path, sep=';')  # Semikolon als Trennzeichen
        
        # Behalte nur die benötigten Spalten
        columns_to_keep = ['firstName', 'lastName', 'linkedinProfileInformation', 'communicationStatus', 'candidateStatus']
        df = df[columns_to_keep]
        
        # Entferne leere Zeilen
        df = df.dropna(how='all')
        
        # Erstelle Namen für bereinigte Datei
        filename, ext = os.path.splitext(input_file)
        clean_filename = f"{filename}_clean{ext}"
        output_path = os.path.join(csv_folder, clean_filename)
        
        # Speichere bereinigte CSV
        df.to_csv(output_path, sep=';', index=False)
        
        print(f"\nBereinigung abgeschlossen:")
        print(f"- Zeilen: {len(df)}")
        print(f"- Spalten: {', '.join(df.columns)}")
        print(f"- Bereinigte Datei gespeichert als: {clean_filename}")
        
        return clean_filename
        
    except Exception as e:
        print(f"Fehler bei der Bereinigung: {str(e)}")
        return None

if __name__ == "__main__":
    clean_filename = clean_csv('CID129 (3).csv')
 