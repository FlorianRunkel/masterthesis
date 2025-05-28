import React from 'react';
import { Box, Typography, Button, CircularProgress } from '@mui/material';
import WorkIcon from '@mui/icons-material/Work';
import SchoolIcon from '@mui/icons-material/School';
import SaveIcon from '@mui/icons-material/Save';

// ProfileDisplay-Komponente: Zeigt die wichtigsten Profildaten eines Kandidaten an
const ProfileDisplay = ({ profile, onSaveCandidate, saving, saveSuccess }) => {
  // Falls kein Profil übergeben wurde, nichts anzeigen
  if (!profile) return null;

  return (
    <Box sx={{ bgcolor: '#fff', borderRadius: '16px', p: '30px', boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)', mb: 4 }}>
      {/* Kopfbereich mit Bild und Basisinfos und Save-Button */}
      <Box className="profile-header" sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '20px', mb: '30px' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
          {/* Profilbild, falls vorhanden */}
          {profile.imageUrl && (
            <Box component="img" src={profile.imageUrl} alt="Profilbild" sx={{ width: '100px', height: '100px', borderRadius: '50%', objectFit: 'cover' }} />
          )}
          {/* Name, aktuelle Position und Standort */}
          <Box className="profile-info">
            <Typography variant="h2" sx={{ fontSize: '1.4rem', fontWeight: 600, color: '#1a1a1a', m: 0, mb: 0.5 }}>{profile.name || 'No name available'}</Typography>
            <Typography sx={{ fontSize: '1rem', color: '#666', mb: 1 }}>{profile.currentTitle || 'No position given'}</Typography>
            <Typography sx={{ fontSize: '1rem', color: '#666' }}>{profile.location || 'No location given'}</Typography>
          </Box>
        </Box>
        {/* Save Candidate Button, nur anzeigen wenn Funktion übergeben */}
        {onSaveCandidate && (
          <Button
            variant="contained"
            startIcon={saving ? <CircularProgress size={10} color="inherit" /> : <SaveIcon />}
            onClick={onSaveCandidate}
            disabled={saving || saveSuccess}
            sx={{
              bgcolor: '#001B41',
              color: '#fff',
              px: 2,
              py: 1.2,
              borderRadius: '8px',
              '&:hover': { bgcolor: '#FF8000' },
              minWidth: '100px',
              fontSize: '0.88rem',
              ml: 2
            }}
          >
            {saving ? 'Save...' : 'Save candidate'}
          </Button>
        )}
      </Box>

      {/* Berufserfahrung und Ausbildung nebeneinander */}
      <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '1fr 1fr' }, gap: 4, mb: 4 }}>
        {/* Berufserfahrung */}
        <Box>
          <Typography variant="h3" sx={{ fontSize: '1.2rem', fontWeight: 700, color: '#1a1a1a', mb: 2, pb: 1, borderBottom: '2px solid #e0e0e0' }}>Work Experiences</Typography>
          {profile.experience && profile.experience.length > 0 ? (
            profile.experience.map((exp, index) => (
              <Box key={index} sx={{ 
                py: 1.5, 
                display: 'flex',
                gap: 2,
                position: 'relative',
                '&::after': index < profile.experience.length - 1 ? {
                  content: '""',
                  position: 'absolute',
                  left: '12px',
                  top: '48px',
                  bottom: '-16px',
                  width: '2px',
                  backgroundColor: '#e0e0e0'
                } : {}
              }}>
                <WorkIcon sx={{ 
                  color: '#FF8000', 
                  fontSize: '1.5rem',
                  mt: 0.5
                }} />
                <Box>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Typography sx={{ fontSize: '1rem', fontWeight: 600, color: '#1a1a1a', mb: 0.2 }}>
                      {exp.title || 'No position given'}
                    </Typography>
                    {exp.endDate === 'Present' && (
                      <Box sx={{
                        display: 'inline-block',
                        bgcolor: '#8AD265',
                        color: '#fff',
                        fontSize: '0.7rem',
                        fontWeight: 600,
                        px: 1.2,
                        py: 0.2,
                        borderRadius: 1,
                        ml: 1,
                        mb: 0.2
                      }}>
                        Current
                      </Box>
                    )}
                  </Box>
                  <Typography sx={{ fontSize: '0.88rem', color: '#666', mb: 0.2 }}>
                    {exp.company || 'No company given'}
                  </Typography>
                </Box>
              </Box>
            ))
          ) : (
            <Typography sx={{ color: '#aaa', fontSize: '0.98rem' }}>No experience given</Typography>
          )}
        </Box>
        {/* Ausbildung */}
        <Box>
          <Typography variant="h3" sx={{ fontSize: '1.2rem', fontWeight: 700, color: '#1a1a1a', mb: 2, pb: 1, borderBottom: '2px solid #e0e0e0' }}>Education</Typography>
          {profile.education && profile.education.length > 0 ? (
            profile.education.map((edu, idx) => (
              <Box key={idx} sx={{ 
                py: 1.5, 
                display: 'flex',
                gap: 2,
                position: 'relative',
                '&::after': idx < profile.education.length - 1 ? {
                  content: '""',
                  position: 'absolute',
                  left: '12px',
                  top: '48px',
                  bottom: '-16px',
                  width: '2px',
                  backgroundColor: '#e0e0e0'
                } : {}
              }}>
                <SchoolIcon sx={{ 
                  color: '#FF8000', 
                  fontSize: '1.5rem',
                  mt: 0.5
                }} />
                <Box>
                  <Typography sx={{ fontSize: '1rem', fontWeight: 600, color: '#1a1a1a', mb: 0.2 }}>{edu.degree || 'No degree given'}</Typography>
                  <Typography sx={{ fontSize: '0.88rem', color: '#666', mb: 0.2 }}>{edu.school || 'No school/university given'}</Typography>
                  <Typography sx={{ fontSize: '0.88rem', color: '#666', opacity: 0.8 }}>{edu.startDate || ''}{(edu.startDate && edu.endDate) ? ' - ' : ''}{edu.endDate || ''}</Typography>
                </Box>
              </Box>
            ))
          ) : (
            <Typography sx={{ color: '#aaa', fontSize: '0.88rem' }}>No education given</Typography>
          )}
        </Box>
      </Box>

      {/* Zusammenfassung, falls vorhanden */}
      {profile.summary && (
        <Box sx={{ mb: 4 }}>
          <Typography variant="h3" sx={{ fontSize: '1.15rem', fontWeight: 600, color: '#1a1a1a', mb: 2, pb: 1, borderBottom: '2px solid #e0e0e0' }}>Summary</Typography>
          <Typography sx={{ color: '#666', lineHeight: 1.6, whiteSpace: 'pre-line' }}>{profile.summary}</Typography>
        </Box>
      )}
    </Box>
  );
};

export default ProfileDisplay; 