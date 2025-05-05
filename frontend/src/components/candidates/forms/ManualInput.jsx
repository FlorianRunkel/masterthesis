import React, { useState } from 'react';
import { Box, Typography, Button, TextField } from '@mui/material';
import PredictionResult from '../prediction/PredictionResult';

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
  const [education, setEducation] = useState([{
    school: '',
    degree: '',
    fieldOfStudy: '',
    startDate: '',
    endDate: ''
  }]);

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

  const handleAddEducation = () => {
    setEducation([...education, {
      school: '',
      degree: '',
      fieldOfStudy: '',
      startDate: '',
      endDate: ''
    }]);
  };

  const handleEducationChange = (index, field, value) => {
    const newEducation = [...education];
    newEducation[index] = {
      ...newEducation[index],
      [field]: value
    };
    setEducation(newEducation);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setPrediction(null);

    try {
      const formData = {
        workExperience: experiences.map(exp => ({
          company: exp.company || "",
          position: exp.position || "",
          startDate: exp.startDate || "",
          endDate: exp.endDate || "Present",
          type: "fullTime",
          location: "",
          description: ""
        })),
        education: education.map(edu => ({
          school: edu.school,
          degree: edu.degree,
          fieldOfStudy: edu.fieldOfStudy,
          startDate: edu.startDate,
          endDate: edu.endDate
        })),
        modelType: selectedModel.toLowerCase()
      };

      // Erstelle das Profil im gleichen Format wie beim Batch-Upload
      const profile_data = {
        firstName: formData.firstName || "Unbekannt",
        lastName: formData.lastName || "Unbekannt",
        profileLink: "",
        modelType: selectedModel.toLowerCase(),
        linkedinProfileInformation: JSON.stringify({
          firstName: formData.firstName || "Unbekannt",
          lastName: formData.lastName || "Unbekannt",
          workExperience: experiences.map(exp => ({
            company: exp.company || "",
            position: exp.position || "",
            startDate: exp.startDate || "",
            endDate: exp.endDate || "Present",
            type: "fullTime",
            location: "",
            description: ""
          })),
          education: education.map(edu => ({
            school: edu.school,
            degree: edu.degree,
            fieldOfStudy: edu.fieldOfStudy,
            startDate: edu.startDate,
            endDate: edu.endDate
          })),
          skills: [],
          location: "",
          headline: "",
          languageSkills: {}
        })
      };

      console.log("Sende Daten:", profile_data); // Debug-Log

      const response = await fetch('http://localhost:5100/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(profile_data),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Fehler bei der Vorhersage');
      }

      const data = await response.json();
      setPrediction(data);
    } catch (err) {
      console.error("Fehler:", err); // Debug-Log
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
            Ausbildung
          </Typography>

          <Box id="education" sx={{ width: '100%', mb: 3 }}>
            {education.map((edu, index) => (
              <Box key={index} sx={{ mb: 2 }}>
                <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr' }, gap: 2 }}>
                  <TextField
                    label="Schule/Hochschule"
                    value={edu.school}
                    onChange={(e) => handleEducationChange(index, 'school', e.target.value)}
                    sx={{
                      '& .MuiOutlinedInput-root': {
                        fontSize: '1.1rem',
                        minHeight: '48px',
                        padding: '5px 0',
                      },
                      input: {
                        fontSize: '1.1rem',
                        padding: '14px 12px'
                      }
                    }}
                  />
                  <TextField
                    label="Abschluss"
                    value={edu.degree}
                    onChange={(e) => handleEducationChange(index, 'degree', e.target.value)}
                    sx={{
                      '& .MuiOutlinedInput-root': {
                        fontSize: '1.1rem',
                        minHeight: '48px',
                        padding: '5px 0',
                      },
                      input: {
                        fontSize: '1.1rem',
                        padding: '14px 12px'
                      }
                    }}
                  />
                  <TextField
                    label="Studienfach"
                    value={edu.fieldOfStudy}
                    onChange={(e) => handleEducationChange(index, 'fieldOfStudy', e.target.value)}
                    sx={{
                      '& .MuiOutlinedInput-root': {
                        fontSize: '1.1rem',
                        minHeight: '48px',
                        padding: '5px 0',
                      },
                      input: {
                        fontSize: '1.1rem',
                        padding: '14px 12px'
                      }
                    }}
                  />
                </Box>
                <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr' }, gap: 2, mt: 2 }}>
                  <TextField
                    label="Startdatum"
                    type="date"
                    value={edu.startDate}
                    onChange={(e) => handleEducationChange(index, 'startDate', e.target.value)}
                    InputLabelProps={{ shrink: true }}
                    helperText={!edu.startDate ? "Bitte Startdatum wählen" : ""}
                    fullWidth
                    sx={{
                      '& .MuiOutlinedInput-root': {
                        fontSize: '1.1rem',
                        minHeight: '48px',
                        padding: '5px 0',
                      },
                      input: {
                        fontSize: '1.1rem',
                        padding: '14px 12px'
                      }
                    }}
                  />
                  <Box sx={{ display: 'flex', gap: 2 }}>
                    <TextField
                      label="Enddatum"
                      type={edu.endDate === 'Present' ? 'text' : 'date'}
                      value={edu.endDate}
                      onChange={(e) => handleEducationChange(index, 'endDate', e.target.value)}
                      InputLabelProps={{ shrink: true }}
                      helperText={!edu.endDate ? "Bitte Enddatum wählen" : ""}
                      fullWidth
                      sx={{
                        '& .MuiOutlinedInput-root': {
                          fontSize: '1.1rem',
                          minHeight: '48px',
                          padding: '5px 0',
                        },
                        input: {
                          fontSize: '1.1rem',
                          padding: '14px 12px'
                        }
                      }}
                    />
                    <Button
                      variant={edu.endDate === 'Present' ? 'contained' : 'outlined'}
                      onClick={() => handleEducationChange(index, 'endDate', edu.endDate === 'Present' ? '' : 'Present')}
                      sx={{ minWidth: 80, minHeight: '48px', height: '57px', p: 0, fontSize: '0.9rem' }}
                    >
                      Present
                    </Button>
                  </Box>
                </Box>
                {index > 0 && (
                  <Button
                    onClick={() => {
                      const newEducation = education.filter((_, i) => i !== index);
                      setEducation(newEducation);
                    }}
                    sx={{
                      minWidth: '100px',
                      padding: '8px 16px',
                      borderRadius: '8px',
                      border: 'none',
                      bgcolor: '#f8f9fa',
                      color: '#666',
                      fontSize: '0.9rem',
                      cursor: 'pointer',
                      transition: 'all 0.3s ease',
                      mt: 1,
                      mb: 1,
                      '&:hover': {
                        bgcolor: '#dc3545',
                        color: 'white'
                      }
                    }}
                  >
                    Entfernen
                  </Button>
                )}
                {index < education.length - 1 && (
                  <Box sx={{ borderBottom: '1px solid #e0e0e0', my: 2 }} />
                )}
              </Box>
            ))}
            <Button
              onClick={handleAddEducation}
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
              WEITERE AUSBILDUNG HINZUFÜGEN
            </Button>
          </Box>

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
              <Box key={index} sx={{ mb: 2 }}>
                <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr' }, gap: 2 }}>
                  <TextField
                    label="Firma"
                    value={exp.company}
                    onChange={e => handleExperienceChange(index, 'company', e.target.value)}
                    sx={{
                      '& .MuiOutlinedInput-root': {
                        fontSize: '1.1rem',
                        minHeight: '48px',
                        padding: '5px 0',
                      },
                      input: {
                        fontSize: '1.1rem',
                        padding: '14px 12px'
                      }
                    }}
                  />
                  <TextField
                    label="Position"
                    value={exp.position}
                    onChange={e => handleExperienceChange(index, 'position', e.target.value)}
                    sx={{
                      '& .MuiOutlinedInput-root': {
                        fontSize: '1.1rem',
                        minHeight: '48px',
                        padding: '5px 0',
                      },
                      input: {
                        fontSize: '1.1rem',
                        padding: '14px 12px'
                      }
                    }}
                  />
                </Box>
                <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr' }, gap: 2, mt: 2 }}>
                  <TextField
                    label="Startdatum"
                    type="date"
                    value={exp.startDate}
                    onChange={e => handleExperienceChange(index, 'startDate', e.target.value)}
                    InputLabelProps={{ shrink: true }}
                    helperText={!exp.startDate ? "Bitte Startdatum wählen" : ""}
                    fullWidth
                    sx={{
                      '& .MuiOutlinedInput-root': {
                        fontSize: '1.1rem',
                        minHeight: '48px',
                        padding: '5px 0',
                      },
                      input: {
                        fontSize: '1.1rem',
                        padding: '14px 12px'
                      }
                    }}
                  />
                  <Box sx={{ display: 'flex', gap: 2 }}>
                    <TextField
                      label="Enddatum"
                      type={exp.endDate === 'Present' ? 'text' : 'date'}
                      value={exp.endDate}
                      onChange={e => handleExperienceChange(index, 'endDate', e.target.value)}
                      InputLabelProps={{ shrink: true }}
                      helperText={!exp.endDate ? "Bitte Enddatum wählen" : ""}
                      fullWidth
                      sx={{
                        '& .MuiOutlinedInput-root': {
                          fontSize: '1.1rem',
                          minHeight: '48px',
                          padding: '5px 0',
                        },
                        input: {
                          fontSize: '1.1rem',
                          padding: '14px 12px'
                        }
                      }}
                    />
                    <Button
                      variant={exp.endDate === 'Present' ? 'contained' : 'outlined'}
                      onClick={() => handleExperienceChange(index, 'endDate', exp.endDate === 'Present' ? '' : 'Present')}
                      sx={{ minWidth: 80, minHeight: '48px', height: '57px', p: 0, fontSize: '0.9rem' }}
                    >
                      Present
                    </Button>
                  </Box>
                </Box>
                {index > 0 && (
                  <Button
                    onClick={() => handleRemoveExperience(index)}
                    sx={{
                      minWidth: '100px',
                      padding: '8px 16px',
                      borderRadius: '8px',
                      border: 'none',
                      bgcolor: '#f8f9fa',
                      color: '#666',
                      fontSize: '0.9rem',
                      cursor: 'pointer',
                      transition: 'all 0.3s ease',
                      mt: 1,
                      mb: 1,
                      '&:hover': {
                        bgcolor: '#dc3545',
                        color: 'white'
                      }
                    }}
                  >
                    Entfernen
                  </Button>
                )}
                {index < experiences.length - 1 && (
                  <Box sx={{ borderBottom: '1px solid #e0e0e0', my: 2 }} />
                )}
              </Box>
            ))}
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
          </Box>

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