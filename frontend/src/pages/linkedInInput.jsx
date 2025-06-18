import React, { useState, useRef, useEffect } from 'react';
import { Box, Typography, TextField, CircularProgress, Button, FormControl} from '@mui/material';
import ProfileDisplay from '../components/display/profile_display';
import PredictionResultClassification from '../components/prediction/prediction_classification';
import PredictionResultTime from '../components/prediction/prediction_time';
import { useTheme } from '@mui/material/styles';
import { useMediaQuery } from '@mui/material';

const LinkedInInput = () => {
  const [linkedinUrl, setLinkedinUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [profileData, setProfileData] = useState(null);
  const [predictionData, setPredictionData] = useState(null);
  const [saving, setSaving] = useState(false);
  const [saveSuccess, setSaveSuccess] = useState(false);
  const [selectedModel, setSelectedModel] = useState(''); // Standardwert z.B. GRU
  const [showModelChangeHint, setShowModelChangeHint] = useState(false);
  const [predictionModelType, setPredictionModelType] = useState('');
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const profileRef = useRef(null);

  useEffect(() => {
    if (profileData && profileRef.current) {
      profileRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [profileData]);

  useEffect(() => {
    if (predictionData) {
      localStorage.setItem('linkedinPrediction', JSON.stringify(predictionData));
      localStorage.setItem('linkedinPredictionModelType', predictionModelType);
    }
  }, [predictionData, predictionModelType]);

  useEffect(() => {
    const saved = localStorage.getItem('linkedinPrediction');
    const savedType = localStorage.getItem('linkedinPredictionModelType');
    if (saved) {
      setPredictionData(JSON.parse(saved));
      setPredictionModelType(savedType || '');
    }
  }, []);

  useEffect(() => {
    const savedProfile = localStorage.getItem('linkedinProfileData');
    if (savedProfile) {
      setProfileData(JSON.parse(savedProfile));
    }
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setPredictionData(null);
    setProfileData(null);
    setLoading(true);
    setError(null);
    setShowModelChangeHint(false);
    setPredictionModelType(selectedModel);

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
      localStorage.setItem('linkedinProfileData', JSON.stringify(profile));

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
      const user = JSON.parse(localStorage.getItem('user'));
      const uid = user?.uid;

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
        headers: {
          'Content-Type': 'application/json',
          'X-User-Uid': uid,
        },
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

  const handleModelChange = (value) => {
    setSelectedModel(value);
    setShowModelChangeHint(true);
  };

  return (
    <Box sx={{ maxWidth: '1200px',  marginLeft: isMobile ? 0 : '240px' }}>
      <Typography variant="h1" sx={{ 
        fontSize: isMobile ? '1.8rem' : '2.5rem', 
        fontWeight: 700, 
        color: '#001242', 
        mb: 2 
      }}>
        LinkedIn Prediction
      </Typography>
      <Typography sx={{ 
        color: '#666', 
        mb: 4, 
        fontSize: isMobile ? '0.9rem' : '1rem', 
        maxWidth: '800px' 
      }}>
        Enter a LinkedIn profile link to automatically create a career prediction based on the available work experience.
      </Typography>

      <Box sx={{ 
        bgcolor: '#fff', 
        borderRadius: '16px', 
        p: '30px', 
        boxShadow: '0 2px 8px rgba(0,0,0,0.08)', 
        mb: 4 
      }}>
       <Typography variant="h2" sx={{ fontSize: '1.36rem', fontWeight: 700, color: '#001242', mb: 0.8 }}>LinkedIn Profile</Typography>
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
                minHeight: '40px',
                padding: '8px 16px',
                '& fieldset': { borderColor: '#e0e0e0', borderWidth: 1 },
                '&:hover fieldset': { borderColor: '#001242' },
                '&.Mui-focused fieldset': { borderColor: '#001242' }
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
      <Box sx={{ bgcolor: '#fff', borderRadius: '14px', p: '32px 24px', boxShadow: '0 2px 8px rgba(0,0,0,0.08)', mb: 1.6 }}>
        <Typography variant="h2" sx={{ fontSize: '1.36rem', fontWeight: 700, color: '#001242', mb: 0.8 }}>
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
            <Box key={option.value} onClick={() => handleModelChange(option.value)} sx={{ cursor: 'pointer', bgcolor: '#fff', border: selectedModel === option.value ? '2px solid #EB7836' : '1.2px solid #e3e6f0', borderRadius: '12.8px', p: 2.4, boxShadow: selectedModel === option.value ? '0 2px 8px rgba(59,71,250,0.08)' : 'none', transition: 'all 0.2s', display: 'flex', flexDirection: 'column', outline: selectedModel === option.value ? '2px solid #EB7836' : 'none', mb: 0.8 }}>
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
            <Box sx={{ bgcolor: '#FFF8E1', border: '1px solid #FFD54F', color: '#EB7836', p: 2, borderRadius: 2, mb: 1, fontSize: '0.8rem'}}>
              Please click 'Start prediction' to run the new model.
            </Box>
          )}
      </Box>
      <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 2, mb: 4 }}>
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
            background: 'linear-gradient(90deg, #EB7836 0%, #EB7836 100%)',
            boxShadow: '0 4px 16px rgba(108,99,255,0.10)',
            textTransform: 'none',
            letterSpacing: 0.16,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 1.2,
            mt: 1.6,
            mb: 2,
            '&:hover': {
              background: 'linear-gradient(90deg, #EB7836 0%, #EB7836 100%)',
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
      {loading && (<Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', my: 4 }}><CircularProgress size={40} thickness={4} sx={{ color: '#001B41' }} /></Box>)}
      {error && (<Box sx={{ bgcolor: '#FEE2E2', border: '1px solid #FCA5A5', color: '#FF2525', p: 3, borderRadius: 2, mb: 3 }}><Typography variant="h6" sx={{ mb: 1 }}>Error</Typography><Typography>{error}</Typography><Box component="ul" sx={{ mt: 2, pl: 2 }}><li>Make sure the URL is correct</li><li>The profile must be publicly accessible</li><li>Try again later</li></Box></Box>)}
      {profileData && (
        <div ref={profileRef}>
          <ProfileDisplay
            profile={profileData}
            onSaveCandidate={handleSaveCandidate}
            saving={saving}
            saveSuccess={saveSuccess}
          />
        </div>
      )}
      {predictionData && (
        <>
          {(predictionModelType === 'tft' || predictionModelType === 'gru') ? (
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