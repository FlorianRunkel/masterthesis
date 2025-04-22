import React, { useState } from 'react';
import { Box, Typography, Button } from '@mui/material';
import PredictionResult from './PredictionResult';

const ManualInput = () => {
  const [experiences, setExperiences] = useState([{
    company: '',
    position: '',
    startDate: '',
    endDate: ''
  }]);
  const [selectedModel, setSelectedModel] = useState('');
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleAddExperience = () => {
    setExperiences([...experiences, {
      company: '',
      position: '',
      startDate: '',
      endDate: ''
    }]);
  };

  const handleRemoveExperience = (index) => {
    if (experiences.length > 1) {
      const newExperiences = experiences.filter((_, i) => i !== index);
      setExperiences(newExperiences);
    }
  };

  const handleExperienceChange = (index, field, value) => {
    const newExperiences = [...experiences];
    newExperiences[index] = {
      ...newExperiences[index],
      [field]: value
    };
    setExperiences(newExperiences);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setPrediction(null);

    try {
      const formData = {
        experiences: experiences.map(exp => ({
          company: exp.company,
          position: exp.position,
          startDate: exp.startDate,
          endDate: exp.endDate || null
        })),
        modelType: selectedModel.toLowerCase()
      };

      const response = await fetch('/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Fehler bei der Vorhersage');
      }

      const data = await response.json();
      setPrediction(data);
    } catch (err) {
      setError(err.message);
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
        Manuelle-Prognose
      </Typography>

      <Typography sx={{
        color: '#666',
        mb: 4,
        fontSize: '1rem',
        maxWidth: '800px'
      }}>
        Analysieren Sie die Wechselwahrscheinlichkeit eines einzelnen Kandidaten basierend auf dessen Berufserfahrung.
      </Typography>

      <Box 
        component="form" 
        onSubmit={handleSubmit}
        sx={{ width: '100%' }}
      >
        <Box
          sx={{
            bgcolor: '#fff',
            borderRadius: '16px',
            p: '30px',
            boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
            mb: 4,
            width: '100%'
          }}
        >
          <Typography variant="h2" sx={{
            fontSize: '1.5rem',
            fontWeight: 600,
            color: '#1a1a1a',
            mb: 3
          }}>
            Berufserfahrung
          </Typography>

          <Box id="experiences" sx={{ width: '100%', mb: 3 }}>
            {experiences.map((exp, index) => (
              <Box
                key={index}
                sx={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(4, 1fr)',
                  gap: '12px',
                  mb: '16px',
                  pb: '16px',
                  borderBottom: index < experiences.length - 1 ? '1px solid rgba(0, 0, 0, 0.1)' : 'none',
                  width: '100%'
                }}
              >
                <input
                  type="text"
                  value={exp.company}
                  onChange={(e) => handleExperienceChange(index, 'company', e.target.value)}
                  placeholder="Firma"
                  style={{
                    width: '100%',
                    padding: '14px',
                    borderRadius: '8px',
                    border: '1px solid #e0e0e0',
                    backgroundColor: 'white',
                    fontSize: '1rem',
                    boxSizing: 'border-box',
                    transition: 'all 0.3s ease',
                    outline: 'none'
                  }}
                />
                <input
                  type="text"
                  value={exp.position}
                  onChange={(e) => handleExperienceChange(index, 'position', e.target.value)}
                  placeholder="Position"
                  style={{
                    width: '100%',
                    padding: '14px',
                    borderRadius: '8px',
                    border: '1px solid #e0e0e0',
                    backgroundColor: 'white',
                    fontSize: '1rem',
                    boxSizing: 'border-box',
                    transition: 'all 0.3s ease',
                    outline: 'none'
                  }}
                />
                <input
                  type="date"
                  value={exp.startDate}
                  onChange={(e) => handleExperienceChange(index, 'startDate', e.target.value)}
                  style={{
                    width: '100%',
                    padding: '14px',
                    borderRadius: '8px',
                    border: '1px solid #e0e0e0',
                    backgroundColor: 'white',
                    fontSize: '1rem',
                    boxSizing: 'border-box',
                    transition: 'all 0.3s ease',
                    outline: 'none'
                  }}
                />
                <input
                  type="date"
                  value={exp.endDate}
                  onChange={(e) => handleExperienceChange(index, 'endDate', e.target.value)}
                  style={{
                    width: '100%',
                    padding: '14px',
                    borderRadius: '8px',
                    border: '1px solid #e0e0e0',
                    backgroundColor: 'white',
                    fontSize: '1rem',
                    boxSizing: 'border-box',
                    transition: 'all 0.3s ease',
                    outline: 'none'
                  }}
                />
                {index > 0 && (
                  <Button
                    onClick={() => handleRemoveExperience(index)}
                    sx={{
                      gridColumn: '1 / -1',
                      width: '100px',
                      marginRight: 'auto',
                      padding: '8px 16px',
                      borderRadius: '8px',
                      border: 'none',
                      bgcolor: '#f8f9fa',
                      color: '#666',
                      fontSize: '0.9rem',
                      cursor: 'pointer',
                      transition: 'all 0.3s ease',
                      '&:hover': {
                        bgcolor: '#dc3545',
                        color: 'white'
                      }
                    }}
                  >
                    Entfernen
                  </Button>
                )}
              </Box>
            ))}
          </Box>

          <Button
            onClick={handleAddExperience}
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
              mb: 3,
              '&:hover': {
                bgcolor: '#FF5F00'
              }
            }}
          >
            WEITERE POSITION HINZUFÜGEN
          </Button>

          <Typography variant="h2" sx={{
            fontSize: '1.5rem',
            fontWeight: 600,
            color: '#1a1a1a',
            mb: 3
          }}>
            KI-Modell
          </Typography>

          <Box sx={{ mb: 3 }}>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              style={{
                width: '100%',
                padding: '14px',
                borderRadius: '8px',
                border: '1px solid #e0e0e0',
                backgroundColor: 'white',
                fontSize: '1rem',
                outline: 'none',
                cursor: 'pointer',
                transition: 'all 0.3s ease'
              }}
            >
              <option value="" disabled>Wählen Sie ein Modell</option>
              <option value="gru">Gated Recurrent Unit (GRU)</option>
              <option value="xgboost">Extreme Gradient Boosting (XGBoost)</option>
              <option value="tft">Temporal Fusion Transformer (TFT)</option>
            </select>
          </Box>

          <Button
            type="submit"
            disabled={loading}
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

      {error && (
        <Box sx={{
          bgcolor: '#fff',
          borderRadius: '16px',
          p: '30px',
          boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
          mb: 4,
          color: '#dc3545',
          width: '100%'
        }}>
          <Typography variant="h6" sx={{ mb: 1 }}>Fehler</Typography>
          <Typography>{error}</Typography>
        </Box>
      )}

      {prediction && <PredictionResult prediction={prediction} />}
    </Box>
  );
};

export default ManualInput; 