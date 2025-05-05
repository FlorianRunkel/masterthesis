import React from 'react';
import { Box, Typography } from '@mui/material';

const ProfileDisplay = ({ profile }) => {
  if (!profile) return null;

  return (
    <Box sx={{
      bgcolor: '#fff',
      borderRadius: '16px',
      p: '30px',
      boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
      mb: 4
    }}>
      <Box className="profile-header" sx={{
        display: 'flex',
        alignItems: 'center',
        gap: '20px',
        mb: '30px'
      }}>
        {profile.imageUrl && (
          <Box
            component="img"
            src={profile.imageUrl}
            alt="Profilbild"
            sx={{
              width: '100px',
              height: '100px',
              borderRadius: '50%',
              objectFit: 'cover',
            }}
          />
        )}
        
        <Box className="profile-info">
          <Typography variant="h2" sx={{
            fontSize: '1.5rem',
            fontWeight: 600,
            color: '#1a1a1a',
            m: 0,
            mb: 0.5
          }}>
            {profile.name || 'Kein Name verfügbar'}
          </Typography>
          
          <Typography sx={{
            fontSize: '1.1rem',
            color: '#666',
            mb: 1
          }}>
            {profile.currentTitle || 'Keine Position angegeben'}
          </Typography>
          
          <Typography sx={{
            fontSize: '0.9rem',
            color: '#666'
          }}>
            {profile.location || 'Kein Standort angegeben'}
          </Typography>
        </Box>
      </Box>

      {/* Berufserfahrung und Ausbildung nebeneinander */}
      <Box sx={{
        display: 'grid',
        gridTemplateColumns: { xs: '1fr', md: '1fr 1fr' },
        gap: 4,
        mb: 4
      }}>
        {/* Berufserfahrung */}
        <Box>
          <Typography variant="h3" sx={{
            fontSize: '1.5rem',
            fontWeight: 700,
            color: '#1a1a1a',
            mb: 2,
            pb: 1,
            borderBottom: '2px solid #e0e0e0'
          }}>
            Berufserfahrung
          </Typography>
          {profile.experience && profile.experience.length > 0 ? (
            profile.experience.map((exp, index) => (
              <Box
                key={index}
                sx={{
                  py: 1.5,
                  borderBottom: index < profile.experience.length - 1 ? '1px solid #e0e0e0' : 'none'
                }}
              >
                <Typography sx={{ fontSize: '1.05rem', fontWeight: 600, color: '#1a1a1a', mb: 0.2 }}>
                  {exp.title || 'Keine Position angegeben'}
                </Typography>
                <Typography sx={{ fontSize: '0.98rem', color: '#666', mb: 0.2 }}>
                  {exp.company || 'Kein Unternehmen angegeben'}
                </Typography>
                <Typography sx={{ fontSize: '0.9rem', color: '#666', opacity: 0.8 }}>
                  {exp.duration || 'Kein Zeitraum angegeben'}
                </Typography>
              </Box>
            ))
          ) : (
            <Typography sx={{ color: '#aaa', fontSize: '0.98rem' }}>Keine Berufserfahrung angegeben</Typography>
          )}
        </Box>
        {/* Ausbildung */}
        <Box>
          <Typography variant="h3" sx={{
            fontSize: '1.5rem',
            fontWeight: 700,
            color: '#1a1a1a',
            mb: 2,
            pb: 1,
            borderBottom: '2px solid #e0e0e0'
          }}>
            Ausbildung
          </Typography>
          {profile.education && profile.education.length > 0 ? (
            profile.education.map((edu, idx) => (
              <Box key={idx} sx={{ py: 1.5, borderBottom: idx < profile.education.length - 1 ? '1px solid #e0e0e0' : 'none' }}>
                <Typography sx={{ fontSize: '1.05rem', fontWeight: 600, color: '#1a1a1a', mb: 0.2 }}>
                  {edu.degree || 'Kein Abschluss angegeben'}
                </Typography>
                <Typography sx={{ fontSize: '0.98rem', color: '#666', mb: 0.2 }}>
                  {edu.school || 'Keine Schule/Hochschule angegeben'}
                </Typography>
                <Typography sx={{ fontSize: '0.9rem', color: '#666', opacity: 0.8 }}>
                  {edu.startDate || ''}{(edu.startDate && edu.endDate) ? ' - ' : ''}{edu.endDate || ''}
                </Typography>
              </Box>
            ))
          ) : (
            <Typography sx={{ color: '#aaa', fontSize: '0.98rem' }}>Keine Ausbildung angegeben</Typography>
          )}
        </Box>
      </Box>

      {profile.summary && (
        <Box sx={{ mb: 4 }}>
          <Typography variant="h3" sx={{
            fontSize: '1.15rem',
            fontWeight: 600,
            color: '#1a1a1a',
            mb: 2,
            pb: 1,
            borderBottom: '2px solid #e0e0e0'
          }}>
            Zusammenfassung
          </Typography>
          
          <Typography sx={{
            color: '#666',
            lineHeight: 1.6,
            whiteSpace: 'pre-line'
          }}>
            {profile.summary}
          </Typography>
        </Box>
      )}
    </Box>
  );
};

export default ProfileDisplay; 