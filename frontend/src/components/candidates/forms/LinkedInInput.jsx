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
      <Typography variant="h1" sx={{ fontSize: '2.5rem', fontWeight: 700, color: '#13213C', mb: 2 }}>LinkedIn Prognose</Typography>
      <Typography sx={{ color: '#666', mb: 4, fontSize: '1rem', maxWidth: '800px' }}>Geben Sie einen LinkedIn-Profillink ein, um automatisch eine Karriereprognose basierend auf den verfügbaren Berufserfahrungen zu erstellen.</Typography>
      <Box sx={{ bgcolor: '#fff', borderRadius: '16px', p: '30px', boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)', mb: 4 }}>
       <Typography variant="h2" sx={{ fontSize: '1.36rem', fontWeight: 700, color: '#2C425C', mb: 0.8 }}>LinkedIn Profil</Typography>
       <Typography sx={{ color: '#888', mb: 3.2, fontSize: '0.86rem' }}>
          Geben Sie einen LinkedIn-Profillink ein, um automatisch eine Karriereprognose basierend auf den verfügbaren Berufserfahrungen zu erstellen.
        </Typography>
        <Box component="form" onSubmit={handleSubmit} sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
          <TextField fullWidth value={linkedinUrl} onChange={(e) => setLinkedinUrl(e.target.value)} placeholder="https://www.linkedin.com/in/username" variant="outlined" sx={{ '& .MuiOutlinedInput-root': { bgcolor: '#fff', fontSize: '1.3rem', minHeight: '40px', padding: '5px 0', '& fieldset': { borderColor: '#e0e0e0', borderWidth: 1 }, '&:hover fieldset': { borderColor: '#13213C' }, '&.Mui-focused fieldset': { borderColor: '#13213C' } }, input: { fontSize: '1.3rem', padding: '18px 14px' } }} />
        </Box>
      </Box>
      <Box sx={{ bgcolor: '#fff', borderRadius: '14px', p: '32px 24px', boxShadow: '0 2px 8px rgba(0,0,0,0.04)', mb: 1.6 }}>
        <Typography variant="h2" sx={{ fontSize: '1.36rem', fontWeight: 700, color: '#2C425C', mb: 0.8 }}>
          KI-Modell auswählen
        </Typography>
        <Typography sx={{ color: '#888', mb: 3.2, fontSize: '0.86rem' }}>
          Wählen Sie das passende Modell für eine präzise Prognose.
        </Typography>
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.6, mb: 1.6 }}>
          <FormControl fullWidth sx={{ display: 'none' }} />
          {[
            {
              value: 'gru',
              title: 'Gated Recurrent Unit (GRU)',
              description: 'Sequenzmodell für Zeitreihen und Karriereverläufe'
            },
            {
              value: 'xgboost',
              title: 'Extreme Gradient Boosting (XGBoost)',
              description: 'Leistungsstarkes Machine-Learning-Modell für strukturierte Daten'
            },
            {
              value: 'tft',
              title: 'Temporal Fusion Transformer (TFT)',
              description: 'Modernes Deep-Learning-Modell für komplexe Zeitreihen'
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
            Prognose starten
          </Button>
        </Box>
      </Box>
      {loading && (<Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', my: 4 }}><CircularProgress size={40} thickness={4} sx={{ color: '#001B41' }} /></Box>)}
      {error && (<Box sx={{ bgcolor: '#FEE2E2', border: '1px solid #FCA5A5', color: '#FF2525', p: 3, borderRadius: 2, mb: 3 }}><Typography variant="h6" sx={{ mb: 1 }}>Fehler</Typography><Typography>{error}</Typography><Box component="ul" sx={{ mt: 2, pl: 2 }}><li>Stellen Sie sicher, dass die URL korrekt ist</li><li>Das Profil muss öffentlich zugänglich sein</li><li>Versuchen Sie es später erneut</li></Box></Box>)}
      {profileData && <ProfileDisplay profile={profileData} />}
      {predictionData && (
        <>
          {(selectedModel === 'tft' || selectedModel === 'gru') ? (
            <PredictionResultTime prediction={predictionData} />
          ) : (
            <PredictionResultClassification prediction={predictionData} />
          )}
          <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', mt: 4, mb: 2, gap: 2 }}>
            <Button variant="contained" startIcon={saving ? <CircularProgress size={20} color="inherit" /> : <SaveIcon />} onClick={handleSaveCandidate} disabled={saving || saveSuccess} sx={{ bgcolor: '#001B41', color: '#fff', px: 4, py: 1.5, borderRadius: '8px', '&:hover': { bgcolor: '#FF8000' }, minWidth: '250px' }}>
              {saving ? 'Speichere...' : 'Kandidat speichern'}
            </Button>
            <Fade in={saveSuccess}>
              <Alert icon={<CheckCircleOutlineIcon fontSize="inherit" />} severity="success" sx={{ mt: 2, bgcolor: '#ECFDF5', color: '#059669', border: '1px solid #A7F3D0', '& .MuiAlert-icon': { color: '#059669' }, borderRadius: '8px', minWidth: '250px' }}>
                Kandidat wurde erfolgreich gespeichert!
              </Alert>
            </Fade>
          </Box>
        </>
      )}
    </Box>
  );
};

export default LinkedInInput; 