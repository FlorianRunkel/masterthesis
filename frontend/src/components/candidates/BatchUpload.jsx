import React, { useState } from 'react';
import { Box, Typography, Button } from '@mui/material';
import ResultsTable from './ResultsTable';

const BatchUpload = () => {
  const [file, setFile] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (!selectedFile) return;
    
    if (!selectedFile.name.endsWith('.csv')) {
      alert("Bitte wählen Sie eine gültige CSV-Datei aus.");
      return;
    }
    
    setFile(selectedFile);
  };

  const handleUpload = async () => {
    if (!file) {
      alert("Bitte wählen Sie eine CSV-Datei aus.");
      return;
    }
    
    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);
    
    try {
      const response = await fetch('http://localhost:5100/predict-batch', {
        method: 'POST',
        body: formData
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      
      if (data.error) {
        setResults({
          error: data.error,
          message: "Bitte überprüfen Sie das Format Ihrer CSV-Datei."
        });
        return;
      }
      
      setResults(data.results);
    } catch (error) {
      console.error('Fehler beim Upload:', error);
      setResults({
        error: error.message,
        message: "Bitte stellen Sie sicher, dass Ihre CSV-Datei folgende Spalten enthält:",
        requirements: [
          "firstName (Vorname)",
          "lastName (Nachname)",
          "linkedinProfile (LinkedIn-URL)",
          "positions (Berufserfahrungen im JSON-Format)"
        ]
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box sx={{ maxWidth: '1200px', margin: '0 auto' }}>
      <Typography variant="h1" sx={{
        fontSize: '2.5rem',
        fontWeight: 700,
        color: '#1a1a1a',
        mb: 2
      }}>
        Batch-Prognose
      </Typography>

      <Typography sx={{
        color: '#666',
        mb: 4,
        fontSize: '1rem',
        maxWidth: '800px'
      }}>
        Laden Sie eine CSV-Datei hoch, um die Wechselwahrscheinlichkeit
        mehrerer Kandidaten gleichzeitig zu analysieren.
      </Typography>

      <Box
        sx={{
          bgcolor: '#fff',
          borderRadius: '16px',
          p: '30px',
          boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
          mb: 4
        }}
      >
        <Typography variant="h2" sx={{
          fontSize: '1.5rem',
          fontWeight: 600,
          color: '#1a1a1a',
          mb: 3
        }}>
          CSV-Datei hochladen
        </Typography>

        <Box
          sx={{
            display: 'flex',
            flexDirection: 'column',
            gap: 3
          }}
        >
          <input
            type="file"
            id="csvFile"
            accept=".csv"
            onChange={handleFileChange}
            style={{ display: 'none' }}
          />
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1, width: '100%' }}>
            <Button
              component="label"
              htmlFor="csvFile"
              variant="outlined"
              sx={{
                width: '100%',
                bgcolor: '#001B41',
                color: 'white',
                border: 'none',
                p: '14px',
                borderRadius: '8px',
                fontSize: '1rem',
                fontWeight: 600,
                cursor: 'pointer',
                transition: 'all 0.3s ease',
                textTransform: 'none',
                '&:hover': {
                  bgcolor: '#FF5F00'
                }
              }}
            >
              DATEI AUSWÄHLEN
            </Button>
            <Typography sx={{ fontSize: '0.9rem', color: '#666', textAlign: 'center' }}>
              {file ? file.name : 'Keine ausgewählt'}
            </Typography>
          </Box>

          <Button
            onClick={handleUpload}
            disabled={!file || loading}
            sx={{
              width: '100%',
              bgcolor: '#001B41',
              color: 'white',
              border: 'none',
              p: '14px',
              borderRadius: '8px',
              fontSize: '1rem',
              fontWeight: 600,
              cursor: 'pointer',
              transition: 'all 0.3s ease',
              textTransform: 'none',
              '&:hover': {
                bgcolor: '#FF5F00'
              },
              '&.Mui-disabled': {
                bgcolor: '#f1f3f4',
                color: '#80868b'
              }
            }}
          >
            PROGNOSE ERSTELLEN
          </Button>
        </Box>
      </Box>

      {loading && (
        <Box sx={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          my: 4
        }}>
          <Box 
            sx={{
              border: '3px solid #f3f3f3',
              borderTop: '3px solid #FF5F00',
              borderRadius: '50%',
              width: '40px',
              height: '40px',
              animation: 'spin 1s linear infinite',
              '@keyframes spin': {
                '0%': { transform: 'rotate(0deg)' },
                '100%': { transform: 'rotate(360deg)' }
              }
            }}
          />
        </Box>
      )}

      {results && <ResultsTable results={results} />}
    </Box>
  );
};

export default BatchUpload; 