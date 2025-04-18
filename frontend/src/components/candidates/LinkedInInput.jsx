import React, { useState } from 'react';
import { Box, Typography, TextField, CircularProgress } from '@mui/material';
import ProfileDisplay from './ProfileDisplay';
import PredictionResult from './PredictionResult';

const LinkedInInput = () => {
  const [linkedinUrl, setLinkedinUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [profileData, setProfileData] = useState(null);
  const [predictionData, setPredictionData] = useState(null);

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

      // Karriere-Vorhersage
      const careerHistory = profile.experience.map(exp => ({
        position: exp.title,
        company: exp.company,
        startDate: exp.duration.split(' - ')[0],
        endDate: exp.duration.split(' - ')[1] === 'Present' ? null : exp.duration.split(' - ')[1]
      }));

      const predictionResponse = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          experiences: careerHistory,
          modelType: 'tft'
        })
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

  return (
    <Box sx={{ maxWidth: '1200px', margin: '0 auto' }}>
      <Typography variant="h1" sx={{
        fontSize: '2.5rem',
        fontWeight: 700,
        color: '#1a1a1a',
        mb: 2
      }}>
        LinkedIn Analyse
      </Typography>

      <Typography sx={{
        color: '#666',
        mb: 4,
        fontSize: '1rem',
        maxWidth: '800px'
      }}>
        Geben Sie einen LinkedIn-Profillink ein, um automatisch eine Karriereprognose basierend auf den verfügbaren Berufserfahrungen zu erstellen.
      </Typography>

      <Box sx={{
        bgcolor: '#fff',
        borderRadius: '16px',
        p: '30px',
        boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
        mb: 4
      }}>
        <Typography variant="h2" sx={{
          fontSize: '1.5rem',
          fontWeight: 600,
          color: '#1a1a1a',
          mb: 3
        }}>
          LinkedIn Profil
        </Typography>

        <Box component="form" onSubmit={handleSubmit} sx={{
          display: 'flex',
          flexDirection: 'column',
          gap: 3
        }}>
          <TextField
            fullWidth
            value={linkedinUrl}
            onChange={(e) => setLinkedinUrl(e.target.value)}
            placeholder="https://www.linkedin.com/in/username"
            variant="outlined"
            sx={{
              '& .MuiOutlinedInput-root': {
                bgcolor: '#fff',
                '& fieldset': {
                  borderColor: '#e0e0e0',
                  borderWidth: 1
                },
                '&:hover fieldset': {
                  borderColor: '#1a1a1a'
                },
                '&.Mui-focused fieldset': {
                  borderColor: '#1a1a1a'
                }
              }
            }}
          />

          <button
            type="submit"
            disabled={loading}
            style={{
              width: '100%',
              background: '#001B41',
              color: '#fff',
              border: 'none',
              padding: '14px',
              borderRadius: '8px',
              fontSize: '1rem',
              fontWeight: 600,
              cursor: 'pointer',
              transition: 'all 0.3s ease'
            }}
          >
            PROGNOSE ERSTELLEN
          </button>
        </Box>
      </Box>

      {loading && (
        <Box sx={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          my: 4
        }}>
          <CircularProgress
            size={40}
            thickness={4}
            sx={{
              color: '#001B41'
            }}
          />
        </Box>
      )}

      {error && (
        <Box sx={{
          bgcolor: '#FEE2E2',
          border: '1px solid #FCA5A5',
          color: '#DC2626',
          p: 3,
          borderRadius: 2,
          mb: 3
        }}>
          <Typography variant="h6" sx={{ mb: 1 }}>Fehler</Typography>
          <Typography>{error}</Typography>
          <Box component="ul" sx={{ mt: 2, pl: 2 }}>
            <li>Stellen Sie sicher, dass die URL korrekt ist</li>
            <li>Das Profil muss öffentlich zugänglich sein</li>
            <li>Versuchen Sie es später erneut</li>
          </Box>
        </Box>
      )}

      {profileData && <ProfileDisplay profile={profileData} />}
      {predictionData && <PredictionResult prediction={predictionData} />}
    </Box>
  );
};

export default LinkedInInput; 