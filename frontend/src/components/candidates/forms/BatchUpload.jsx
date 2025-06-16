import React, { useState } from 'react';
import { Box, Typography, Button, Alert } from '@mui/material';
import ResultsTableClassification from '../display/ResultsTableClassification';
import LoadingSpinner from '../../common/LoadingSpinner';
import ResultsTableTimeSeries from '../display/ResultsTableTimeSeries';
import { useTheme } from '@mui/material/styles';
import { useMediaQuery } from '@mui/material';

const modelOptions = [
  {
    value: 'gru',
    title: 'Gated Recurrent Unit (GRU)',
    description: 'Sequence model for time series and career trajectories'
  },
  {
    value: 'xgboost',
    title: 'Extreme Gradient Boosting (XGBoost)',
    description: 'Powerful machine learning model for structured data'
  },
  {
    value: 'tft',
    title: 'Temporal Fusion Transformer (TFT)',
    description: 'Modern deep learning model for complex time series'
  }
];

const BatchUpload = () => {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [results, setResults] = useState(null);
  const [isSaving, setIsSaving] = useState(false);
  const [saveSuccess, setSaveSuccess] = useState(false);
  const [saveError, setSaveError] = useState(null);
  const [modelType, setModelType] = useState('');
  const [originalProfiles, setOriginalProfiles] = useState([]);
  const [showModelChangeHint, setShowModelChangeHint] = useState(false);
  const [resultsModelType, setResultsModelType] = useState('');
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (!selectedFile) return;
    if (!selectedFile.name.endsWith('.csv')) {
      alert("Please select a valid CSV file.");
      return;
    }
    setFile(selectedFile);
  };

  const handleUpload = async () => {
    setShowModelChangeHint(false);
    if (!file) {
      alert("Please select a CSV file.");
      return;
    }
    setResults(null);
    setOriginalProfiles([]);
    setResultsModelType(modelType);
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
        throw new Error('Upload failed');
      }
      const data = await response.json();
      if (data.error) {
        setResults({
          error: data.error,
          message: "Please check the format of your CSV file."
        });
        return;
      }
      setResults(data.results);
      if (data.originalProfiles) setOriginalProfiles(data.originalProfiles);
    } catch (error) {
      setError(error.message);
      setResults({
        error: error.message,
        message: "Please make sure your CSV file contains the following columns:",
        requirements: [
          "firstName (first name)",
          "lastName (last name)",
          "linkedinProfile (LinkedIn-URL)",
          "positions (experience in JSON format)"
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
      // Hole für alle Kandidaten mit LinkedIn-Link die Profildaten
      const candidatesWithProfile = await Promise.all(
        candidates.map(async (candidate) => {
          if (candidate.linkedinProfile) {
            try {
              const response = await fetch('http://localhost:5100/scrape-linkedin', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ url: candidate.linkedinProfile }),
              });
              const data = await response.json();
              if (data && !data.error) {
                // Profildaten extrahieren und auf oberster Ebene speichern
                const [firstName, ...rest] = (data.name || '').split(' ');
                const lastName = rest.join(' ');
                return {
                  ...candidate,
                  firstName: firstName || '',
                  lastName: lastName || '',
                  currentPosition: data.currentTitle || '',
                  imageUrl: data.imageUrl || '',
                  experience: data.experience || [],
                  location: data.location || '',
                  industry: data.industry || '',
                  linkedinProfileInformation: JSON.stringify(data),
                };
              }
            } catch (err) {
              // Fehler beim Scrapen: Kandidat trotzdem speichern, aber mit Fehlerhinweis
              return {
                ...candidate,
                scrapeError: 'LinkedIn scraping failed',
              };
            }
          }
          // Falls kein LinkedIn-Link, Kandidat unverändert zurückgeben
          return candidate;
        })
      );

      // Jetzt alle Kandidaten speichern
      const candidatesWithModel = candidatesWithProfile.map(candidate => ({
        ...candidate,
        modelType: modelType
      }));

      const response = await fetch('http://localhost:5100/api/candidates', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(candidatesWithModel),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || 'Error saving candidates');
      }
      setSaveSuccess(true);
      setResults(null); // Reset results after successful save
      setFile(null);
    } catch (error) {
      setSaveError(error.message);
    } finally {
      setIsSaving(false);
    }
  };

  const handleModelChange = (value) => {
    setModelType(value);
    setShowModelChangeHint(true);
  };

  return (
    <Box sx={{ maxWidth: '1200px',  marginLeft: isMobile ? 0 : '240px' }}>

      <Typography variant="h1" sx={{ 
        fontSize: isMobile ? '1.8rem' : '2.5rem', 
        fontWeight: 700, 
        color: '#13213C', 
        mb: 2 
      }}>
        Batch Upload
      </Typography>
      <Typography sx={{ 
        color: '#666', 
        mb: 4, 
        fontSize: isMobile ? '0.9rem' : '1rem', 
        maxWidth: '800px' 
      }}>
        Upload a CSV file to analyze the job change probability of multiple candidates at once.
      </Typography>
      
      {/* Upload-Box */}
      <Box sx={{ bgcolor: '#fff', borderRadius: '12.8px', p: '24px', boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)', mb: 3.2 }}>
        <Typography variant="h2" sx={{ fontSize: '1.2rem', fontWeight: 600, color: '#1a1a1a', mb: 2.4 }}>Upload CSV file</Typography>
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2.4 }}>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.8, width: '100%' }}>
            <Button
              component="label"
              htmlFor="csvFile"
              sx={{
                width: '100%',
                height: '40px',
                bgcolor: '#fff',
                color: '#001B41',
                border: '1.6px dashed #001B41',
                borderRadius: '9.6px',
                fontSize: '0.8rem',
                fontWeight: 700,
                cursor: 'pointer',
                textTransform: 'none',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                letterSpacing: 0.16,
                transition: 'all 0.2s',
                boxShadow: 'none',
                '&:hover': {
                  bgcolor: '#fff',
                  border: '1.6px solid #FF8000',
                  color: '#FF8000',
                },
              }}
            >
              <span style={{ fontWeight: 700, fontSize: '0.8rem', display: 'flex', alignItems: 'center', gap: 6.4 }}>
                SELECT FILE
              </span>
              <input type="file" id="csvFile" accept=".csv" onChange={handleFileChange} style={{ display: 'none' }} />
            </Button>
            <Typography sx={{ fontSize: '0.8rem', color: '#666', textAlign: 'center', paddingTop: '8px' }}>{file ? file.name : 'No file selected'}</Typography>
          </Box>
        </Box>
      </Box>

      {/* Modellauswahl-Box */}
      <Box sx={{ bgcolor: '#fff', borderRadius: '14px', p: '32px 24px', boxShadow: '0 2px 8px rgba(0,0,0,0.08)', mb: 1.6 }}>
        <Typography variant="h2" sx={{ fontSize: '1.36rem', fontWeight: 700, color: '#13213C', mb: 0.8 }}>
          Select AI model
        </Typography>
        <Typography sx={{ color: '#888', mb: 3.2, fontSize: '0.86rem' }}>
          Select the appropriate model for a precise prediction.
        </Typography>
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.6, mb: 1.6 }}>
          {modelOptions.map(option => (
            <Box key={option.value} onClick={() => handleModelChange(option.value)} sx={{ cursor: 'pointer', bgcolor: '#fff', border: modelType === option.value ? '1.6px solid #FF8000' : '1.2px solid #e3e6f0', borderRadius: '12.8px', p: 2.4, boxShadow: modelType === option.value ? '0 2px 8px rgba(59,71,250,0.08)' : 'none', transition: 'all 0.2s', display: 'flex', flexDirection: 'column', outline: modelType === option.value ? '1.6px solid #FF8000' : 'none', mb: 0.8 }}>
              <Typography sx={{ fontWeight: 700, fontSize: '0.94rem', color: '#1a1a1a', mb: 0.4 }}>
                {option.title}
              </Typography>
              <Typography sx={{ color: '#888', fontSize: '0.84rem' }}>
                {option.description}
              </Typography>
            </Box>
          ))}
        </Box>
        {showModelChangeHint && (
            <Box sx={{ bgcolor: '#FFF8E1', border: '1px solid #FFD54F', color: '#FF8000', p: 2, borderRadius: 2, mb: 1, fontSize: '0.8rem'}}>
              Please click 'Start prediction' to run the new model.
            </Box>
          )}
        <Box sx={{ display: 'flex', justifyContent: 'center', mt: 1.6 }}>
          <Button
            onClick={handleUpload}
            disabled={!file || loading || !modelType}
            sx={{
              minWidth: 256,
              px: 3.2,
              py: 1.44,
              fontSize: '0.94rem',
              fontWeight: 700,
              borderRadius: '11.2px',
              color: '#fff',
              background: 'linear-gradient(90deg, #f4a65892 0%, #f4a65892 100%)',
              boxShadow: '0 4px 16px rgba(108,99,255,0.10)',
              textTransform: 'none',
              letterSpacing: 0.16,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: 1.2,
              mt: 1.6,
              mx: 'auto',
              '&:hover': {
                background: 'linear-gradient(90deg, #FF8000 0%, #FF8000 100%)',
              },
              '&.Mui-disabled': {
                background: '#e3e6f0',
                color: '#bdbdbd',
              },
            }}
          >
            Start prediction
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
          Candidates were successfully saved!
        </Alert>
      )}
      {loading && <LoadingSpinner />}
      {results && !loading && (
        <Box sx={{ mt: 3 }}>
          {resultsModelType === 'tft' || resultsModelType === 'gru' ? (
            <ResultsTableTimeSeries
              results={results}
              onSave={handleSaveCandidates}
              isSaving={isSaving}
              originalProfiles={originalProfiles}
            />
          ) : (
            <ResultsTableClassification
              results={results}
              onSave={handleSaveCandidates}
              isSaving={isSaving}
              originalProfiles={originalProfiles}
            />
          )}
        </Box>
      )}
    </Box>
  );
};

export default BatchUpload; 