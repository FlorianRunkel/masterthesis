import React, { useState } from 'react';
import { Box, Typography, Button } from '@mui/material';

const Batch = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    setSelectedFile(file);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!selectedFile) return;

    setIsLoading(true);
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch('http://localhost:5100/predict-batch', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Fehler beim Hochladen der Datei');
      }

      const data = await response.json();
      console.log(data);
    } catch (error) {
      console.error('Fehler:', error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Box sx={{ p: 4 }}>
      <Typography
        variant="h4"
        component="h1"
        sx={{
          color: '#0A1929',
          fontWeight: 700,
          mb: 2
        }}
      >
        Batch-Prognose
      </Typography>

      <Typography
        sx={{
          color: '#666',
          mb: 4,
          fontSize: '1.1rem'
        }}
      >
        Laden Sie eine CSV-Datei hoch, um die Wechselwahrscheinlichkeit
        mehrerer Kandidaten gleichzeitig zu analysieren.
      </Typography>

      <Box
        sx={{
          bgcolor: 'white',
          borderRadius: 4,
          p: 4,
          boxShadow: '0 8px 25px rgba(0, 0, 0, 0.1)',
          maxWidth: '100%'
        }}
      >
        <Typography
          variant="h6"
          sx={{
            color: '#0A1929',
            fontWeight: 600,
            mb: 3
          }}
        >
          CSV-Datei hochladen
        </Typography>

        <Box
          className="file-upload-wrapper"
          sx={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            gap: 2,
            mb: 3
          }}
        >
          <input
            type="file"
            id="csvFile"
            accept=".csv"
            onChange={handleFileChange}
            style={{ display: 'none' }}
          />
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, width: '100%' }}>
            <Typography sx={{ color: '#666', flex: 1 }}>
              Datei auswählen
            </Typography>
            <Button
              component="label"
              htmlFor="csvFile"
              variant="outlined"
              sx={{
                bgcolor: 'white',
                border: '1px solid rgba(0, 0, 0, 0.23)',
                color: 'black',
                textTransform: 'none',
                px: 3,
                minWidth: 'auto',
                '&:hover': {
                  bgcolor: 'rgba(0, 0, 0, 0.04)',
                  border: '1px solid rgba(0, 0, 0, 0.23)'
                }
              }}
            >
              Datei auswählen
            </Button>
            <Typography sx={{ color: '#666', flex: 1, textAlign: 'right' }}>
              {selectedFile ? selectedFile.name : 'Keine ausgewählt'}
            </Typography>
          </Box>

          <Button
            onClick={handleSubmit}
            disabled={!selectedFile || isLoading}
            sx={{
              width: '100%',
              maxWidth: '400px',
              bgcolor: '#4285f4',
              color: 'white',
              textTransform: 'none',
              py: 1.5,
              borderRadius: 2,
              fontWeight: 600,
              '&:hover': {
                bgcolor: '#3367d6'
              },
              '&.Mui-disabled': {
                bgcolor: '#f1f3f4',
                color: '#80868b'
              }
            }}
          >
            {isLoading ? 'Wird verarbeitet...' : 'Prognose erstellen'}
          </Button>
        </Box>
      </Box>
    </Box>
  );
};

export default Batch; 