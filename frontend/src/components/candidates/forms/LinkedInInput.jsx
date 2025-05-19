import React, { useState } from 'react';
import { Box, Typography, TextField, CircularProgress, Button, Alert, Fade, FormControl, InputLabel, Select, MenuItem } from '@mui/material';
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
        if (selectedModel === 'tft') {
          if (startDate && /^\d{4}$/.test(startDate)) {
            startDate = `01/${startDate}`;
          }
          if (endDate && /^\d{4}$/.test(endDate)) {
            endDate = `01/${endDate}`;
          }
          if (!endDate || endDate === 'Present') {
            endDate = 'Present';
          }
        }
      
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
        experience: profileData.experience
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
    <Box sx={{ maxWidth: '1200px', margin: '0 auto' }}>
      <Typography variant="h1" sx={{ fontSize: '2.5rem', fontWeight: 700, color: '#13213C', mb: 2 }}>LinkedIn Prognose</Typography>
      <Typography sx={{ color: '#666', mb: 4, fontSize: '1rem', maxWidth: '800px' }}>Geben Sie einen LinkedIn-Profillink ein, um automatisch eine Karriereprognose basierend auf den verfügbaren Berufserfahrungen zu erstellen.</Typography>
      <Box sx={{ bgcolor: '#fff', borderRadius: '16px', p: '30px', boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)', mb: 4 }}>
        <Typography variant="h2" sx={{ fontSize: '1.5rem', fontWeight: 600, color: '#13213C', mb: 3 }}>LinkedIn Profil</Typography>
        <Box component="form" onSubmit={handleSubmit} sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
          <TextField fullWidth value={linkedinUrl} onChange={(e) => setLinkedinUrl(e.target.value)} placeholder="https://www.linkedin.com/in/username" variant="outlined" sx={{ '& .MuiOutlinedInput-root': { bgcolor: '#fff', fontSize: '1.3rem', minHeight: '40px', padding: '5px 0', '& fieldset': { borderColor: '#e0e0e0', borderWidth: 1 }, '&:hover fieldset': { borderColor: '#13213C' }, '&.Mui-focused fieldset': { borderColor: '#13213C' } }, input: { fontSize: '1.3rem', padding: '18px 14px' } }} />
          <Typography sx={{ fontWeight: 600, fontSize: '1.1rem', mb: 1 }}>Modelltyp auswählen</Typography>
          <FormControl fullWidth sx={{ mb: 2 }}>
            <InputLabel id="model-type-label">Modelltyp</InputLabel>
            <Select labelId="model-type-label" value={selectedModel} label="Modelltyp" onChange={(e) => setSelectedModel(e.target.value)} sx={{ '&:hover': { borderColor: 'transparent' }, '&.Mui-focused': { borderColor: 'transparent' } }}>
              <MenuItem value="gru">Gated Recurrent Units (GRU)</MenuItem>
              <MenuItem value="xgboost">Extrem Gradient Boosting (XGBoost)</MenuItem>
              <MenuItem value="tft">Temporal Fusion Transformer (TFT)</MenuItem>
            </Select>
          </FormControl>
          <button type="submit" disabled={loading} style={{ width: '100%', background: '#13213C', color: '#fff', border: 'none', padding: '14px', borderRadius: '8px', fontSize: '1rem', fontWeight: 600, cursor: 'pointer', transition: 'all 0.3s ease' }}>PROGNOSE ERSTELLEN</button>
        </Box>
      </Box>
      {loading && (<Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', my: 4 }}><CircularProgress size={40} thickness={4} sx={{ color: '#001B41' }} /></Box>)}
      {error && (<Box sx={{ bgcolor: '#FEE2E2', border: '1px solid #FCA5A5', color: '#FF2525', p: 3, borderRadius: 2, mb: 3 }}><Typography variant="h6" sx={{ mb: 1 }}>Fehler</Typography><Typography>{error}</Typography><Box component="ul" sx={{ mt: 2, pl: 2 }}><li>Stellen Sie sicher, dass die URL korrekt ist</li><li>Das Profil muss öffentlich zugänglich sein</li><li>Versuchen Sie es später erneut</li></Box></Box>)}
      {profileData && <ProfileDisplay profile={profileData} />}
      {predictionData && (<>{selectedModel === 'tft' ? (<PredictionResultTime prediction={predictionData} />) : (<PredictionResultClassification prediction={predictionData} />)}<Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', mt: 4, mb: 2, gap: 2 }}><Button variant="contained" startIcon={saving ? <CircularProgress size={20} color="inherit" /> : <SaveIcon />} onClick={handleSaveCandidate} disabled={saving || saveSuccess} sx={{ bgcolor: '#001B41', color: '#fff', px: 4, py: 1.5, borderRadius: '8px', '&:hover': { bgcolor: '#FF8000' }, minWidth: '250px' }}>{saving ? 'Speichere...' : 'Kandidat speichern'}</Button><Fade in={saveSuccess}><Alert icon={<CheckCircleOutlineIcon fontSize="inherit" />} severity="success" sx={{ mt: 2, bgcolor: '#ECFDF5', color: '#059669', border: '1px solid #A7F3D0', '& .MuiAlert-icon': { color: '#059669' }, borderRadius: '8px', minWidth: '250px' }}>Kandidat wurde erfolgreich gespeichert!</Alert></Fade></Box></>)}
    </Box>
  );
};

export default LinkedInInput; 