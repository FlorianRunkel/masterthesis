import React, { useState } from 'react';
import { Box, Typography, Button, Alert, FormControl, InputLabel, Select, MenuItem } from '@mui/material';
import ResultsTableClassification from '../display/ResultsTableClassification';
import LoadingSpinner from '../../common/LoadingSpinner';
import ResultsTableTime from '../display/ResultsTableTime';

const BatchUpload = () => {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error] = useState(null);
  const [results, setResults] = useState(null);
  const [isSaving, setIsSaving] = useState(false);
  const [saveSuccess, setSaveSuccess] = useState(false);
  const [saveError, setSaveError] = useState(null);
  const [modelType, setModelType] = useState('xgboost');

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
    formData.append('modelType', modelType);
    
    try {
      const response = await fetch('http://localhost:5100/predict-batch', {
        method: 'POST',
        body: formData
      });
      
      if (!response.ok) {
        throw new Error('Upload fehlgeschlagen');
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

  const handleSaveCandidates = async (candidates) => {
    setIsSaving(true);
    setSaveError(null);
    setSaveSuccess(false);
    
    try {
      const response = await fetch('http://localhost:5100/api/candidates', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(candidates),
      });
      
      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.error || 'Fehler beim Speichern der Kandidaten');
      }
      
      setSaveSuccess(true);
      setResults(null); // Reset results after successful save
      setFile(null);
      
    } catch (error) {
      console.error('Fehler beim Speichern:', error);
      setSaveError(error.message);
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <Box sx={{
      maxWidth: '1200px',
      margin: '0 auto',
    }}>

      <Typography variant="h1" sx={{
        fontSize: '2.5rem',
        fontWeight: 700,
        color: '#1a1a1a',
        mb: 2
      }}>
        Batch Upload
      </Typography>

      <Typography sx={{
        color: '#666',
        mb: 4,
        fontSize: '1rem',
        maxWidth: '800px'
      }}>Laden Sie eine CSV-Datei hoch, um die Wechselwahrscheinlichkeit
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
            <Typography sx={{ fontSize: '1rem', color: '#666', textAlign: 'center', paddingTop: '10px' }}>
              {file ? file.name : 'Keine ausgewählt'}
            </Typography>
          </Box>

          <Typography sx={{ fontWeight: 600, fontSize: '1.1rem', mb: 1 }}>
              Modelltyp auswählen
          </Typography>

          <FormControl fullWidth sx={{ mb: 2 }}>
            <InputLabel id="model-type-label">Modelltyp</InputLabel>
            <Select
              labelId="model-type-label"
              value={modelType}
              label="Modelltyp"
              onChange={(e) => setModelType(e.target.value)}
            >
              <MenuItem value="xgboost">Gated Recurrent Units (GRU)</MenuItem>
              <MenuItem value="gru">Extrem Gradient Boosting (XGBoost)</MenuItem>
              <MenuItem value="tft">Temporal Fusion Transformer (TFT)</MenuItem>
            </Select>
          </FormControl>
          <input
            type="file"
            id="csvFile"
            accept=".csv"
            onChange={handleFileChange}
            style={{ display: 'none' }}
          />

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

      {error && (
        <Alert severity="error" sx={{ mt: 2 }}>
          {error}
        </Alert>
      )}
      
      {saveError && (
        <Alert severity="error" sx={{ mt: 2 }}>
          {saveError}
        </Alert>
      )}
      
      {saveSuccess && (
        <Alert severity="success" sx={{ mt: 2 }}>
          Kandidaten wurden erfolgreich gespeichert!
        </Alert>
      )}
      
      {loading && <LoadingSpinner />}
      
      {results && !loading && (
        <Box sx={{ mt: 3 }}>
          {modelType === 'tft' ? (
            <ResultsTableTime results={results} />
          ) : (
            <ResultsTableClassification results={results} />
          )}
        </Box>
      )}
    </Box>
  );
};

export default BatchUpload; 