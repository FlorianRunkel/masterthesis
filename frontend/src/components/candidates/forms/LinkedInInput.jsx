import React, { useState } from 'react';
import { Box, Typography, TextField, CircularProgress, Button, Alert, Fade, FormControl} from '@mui/material';
import SaveIcon from '@mui/icons-material/Save';
import CheckCircleOutlineIcon from '@mui/icons-material/CheckCircleOutline';
import ProfileDisplay from '../display/ProfileDisplay';
import PredictionResultClassification from '../prediction/PredictionResultClassification';
import PredictionResultTime from '../prediction/PredictionResultTime';

const LinkedInInput = () => {
  const [linkedinUrl, setLinkedinUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [profileData, setProfileData] = useState(null);
  const [predictionData, setPredictionData] = useState(null);
  const [saving, setSaving] = useState(false);
  const [saveSuccess, setSaveSuccess] = useState(false);
  const [selectedModel, setSelectedModel] = useState('gru'); // Standardwert z.B. GRU

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setProfileData(null);
    setPredictionData(null);

    try {
      // LinkedIn-Profil abrufen
      const profileResponse = await fetch('/scrape-linkedin', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: linkedinUrl })
      });

      if (!profileResponse.ok) {
        const errorData = await profileResponse.json();
        throw new Error(errorData.error || 'Fehler beim Laden des LinkedIn-Profils');
      }

      const profile = await profileResponse.json();
      setProfileData(profile);

      // Namen extrahieren
      const [firstName, ...rest] = profile.name.split(' ');
      const lastName = rest.join(' ');

      // Berufserfahrung aufbereiten
      const workExperience = profile.experience.map(exp => {
        // Versuche, das Datum zu parsen
        let startDate = exp.startDate;
        let endDate = exp.endDate;
      
        // Falls nur Jahr vorhanden, ergänze Monat
      
        return {
          company: exp.company,
          position: exp.title,
          startDate,
          endDate,
          type: "fullTime",
          location: "",
          description: ""
        };
      });

      // Optional: education, skills etc. ergänzen, falls vorhanden
      const education = profile.education || [];

      const profile_data = {
        firstName: firstName || "Unbekannt",
        lastName: lastName || "Unbekannt",
        profileLink: linkedinUrl,
        modelType: selectedModel,
        linkedinProfileInformation: JSON.stringify({
          firstName: firstName || "Unbekannt",
          lastName: lastName || "Unbekannt",
          workExperience,
          education,
          skills: [],
          location: profile.location || "",
          headline: profile.currentTitle || "",
          languageSkills: {}
        })
      };

      // Karriere-Vorhersage
      const predictionResponse = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(profile_data)
      });

      if (!predictionResponse.ok) {
        const errorData = await predictionResponse.json();
        throw new Error(errorData.error || 'Fehler bei der Karriere-Analyse');
      }

      const prediction = await predictionResponse.json();
      setPredictionData(prediction);

    } catch (error) {
      setError(error.message);
    } finally {
      setLoading(false);
    }
  };

  const handleSaveCandidate = async () => {
    if (!profileData || !predictionData) return;
    
    setSaving(true);
    setSaveSuccess(false);
    setError(null);

    try {
      const candidateData = {
        firstName: profileData.name.split(' ')[0],
        lastName: profileData.name.split(' ').slice(1).join(' '),
        linkedinProfile: linkedinUrl,
        currentPosition: profileData.currentTitle,
        location: profileData.location,
        confidence: [predictionData.confidence],
        recommendations: predictionData.recommendations,
        imageUrl: profileData.imageUrl,
        industry: profileData.industry,
        experience: profileData.experience,
        modelType: selectedModel
      };

      const response = await fetch('/api/candidates', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify([candidateData])
      });

      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.error || 'Fehler beim Speichern des Kandidaten');
      }

      if (result.skippedCount > 0) {
        setError('Dieser Kandidat wurde bereits analysiert und ist in der Datenbank gespeichert.');
        return;
      }

      setSaveSuccess(true);
    } catch (error) {
      setError(error.message);
    } finally {
      setSaving(false);
    }
  };

  return (
    <Box sx={{ maxWidth: '1200px', ml: 0 }}>
      <Typography variant="h1" sx={{ fontSize: '2.5rem', fontWeight: 700, color: '#13213C', mb: 2 }}>LinkedIn Prediction</Typography>
      <Typography sx={{ color: '#666', mb: 4, fontSize: '1rem', maxWidth: '800px' }}>Enter a LinkedIn profile link to automatically create a career prediction based on the available work experience.</Typography>
      <Box sx={{ bgcolor: '#fff', borderRadius: '16px', p: '30px', boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)', mb: 4 }}>
       <Typography variant="h2" sx={{ fontSize: '1.36rem', fontWeight: 700, color: '#13213C', mb: 0.8 }}>LinkedIn Profile</Typography>
       <Typography sx={{ color: '#888', mb: 3.2, fontSize: '0.88rem' }}>
          Add the LinkedIn profile link to get a career prediction.
        </Typography>
        <Box component="form" onSubmit={handleSubmit} sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
          <TextField
            fullWidth
            value={linkedinUrl}
            onChange={(e) => setLinkedinUrl(e.target.value)}
            placeholder="https://www.linkedin.com/in/username"
            variant="outlined"
            sx={{
              '& .MuiOutlinedInput-root': {
                bgcolor: '#fff',
                fontSize: '0.88rem',
                minHeight: '30px',
                padding: '0px 0',
                '& fieldset': { borderColor: '#e0e0e0', borderWidth: 1 },
                '&:hover fieldset': { borderColor: '#13213C' },
                '&.Mui-focused fieldset': { borderColor: '#13213C' }
              },
              input: {
                fontSize: '0.88rem',
                padding: '0px 0px',
                '&::placeholder': {
                  fontSize: '0.88rem',
                  color: '#bdbdbd',
                  opacity: 1
                }
              }
            }}
          />
        </Box>
      </Box>
      <Box sx={{ bgcolor: '#fff', borderRadius: '14px', p: '32px 24px', boxShadow: '0 2px 8px rgba(0,0,0,0.04)', mb: 1.6 }}>
        <Typography variant="h2" sx={{ fontSize: '1.36rem', fontWeight: 700, color: '#13213C', mb: 0.8 }}>
          Select AI model
        </Typography>
        <Typography sx={{ color: '#888', mb: 3.2, fontSize: '0.86rem' }}>
          Select the appropriate model for a precise prediction.
        </Typography>
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.6, mb: 1.6 }}>
          <FormControl fullWidth sx={{ display: 'none' }} />
          {[
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
          ].map(option => (
            <Box key={option.value} onClick={() => setSelectedModel(option.value)} sx={{ cursor: 'pointer', bgcolor: '#fff', border: selectedModel === option.value ? '2px solid #FF8000' : '1.2px solid #e3e6f0', borderRadius: '12.8px', p: 2.4, boxShadow: selectedModel === option.value ? '0 2px 8px rgba(59,71,250,0.08)' : 'none', transition: 'all 0.2s', display: 'flex', flexDirection: 'column', outline: selectedModel === option.value ? '2px solid #FF8000' : 'none', mb: 0.8 }}>
              <Typography sx={{ fontWeight: 700, fontSize: '0.94rem', color: '#1a1a1a', mb: 0.4 }}>
                {option.title}
              </Typography>
              <Typography sx={{ color: '#888', fontSize: '0.84rem' }}>
                {option.description}
              </Typography>
            </Box>
          ))}
        </Box>
        <Box sx={{ display: 'flex', justifyContent: 'center', mt: 1.6 }}>
          <Button
            onClick={handleSubmit}
            disabled={loading || !selectedModel || !linkedinUrl}
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
      {loading && (<Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', my: 4 }}><CircularProgress size={40} thickness={4} sx={{ color: '#001B41' }} /></Box>)}
      {error && (<Box sx={{ bgcolor: '#FEE2E2', border: '1px solid #FCA5A5', color: '#FF2525', p: 3, borderRadius: 2, mb: 3 }}><Typography variant="h6" sx={{ mb: 1 }}>Error</Typography><Typography>{error}</Typography><Box component="ul" sx={{ mt: 2, pl: 2 }}><li>Make sure the URL is correct</li><li>The profile must be publicly accessible</li><li>Try again later</li></Box></Box>)}
      {profileData && (
        <ProfileDisplay
          profile={profileData}
          onSaveCandidate={handleSaveCandidate}
          saving={saving}
          saveSuccess={saveSuccess}
        />
      )}
      {predictionData && (
        <>
          {(selectedModel === 'tft' || selectedModel === 'gru') ? (
            <PredictionResultTime prediction={predictionData} />
          ) : (
            <PredictionResultClassification prediction={predictionData} />
          )}
        </>
      )}
    </Box>
  );
};

export default LinkedInInput; 