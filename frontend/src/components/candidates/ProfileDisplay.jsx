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
              objectFit: 'cover'
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

      <Box sx={{ mb: 4 }}>
        <Typography variant="h3" sx={{
          fontSize: '1.3rem',
          fontWeight: 600,
          color: '#1a1a1a',
          mb: 2,
          pb: 1,
          borderBottom: '2px solid #e0e0e0'
        }}>
          Berufserfahrung
        </Typography>

        {profile.experience && profile.experience.map((exp, index) => (
          <Box
            key={index}
            sx={{
              py: 2.5,
              borderBottom: index < profile.experience.length - 1 ? '1px solid #e0e0e0' : 'none'
            }}
          >
            <Typography sx={{
              fontSize: '1.1rem',
              fontWeight: 600,
              color: '#1a1a1a',
              mb: 0.5
            }}>
              {exp.title || 'Keine Position angegeben'}
            </Typography>
            
            <Typography sx={{
              fontSize: '1rem',
              color: '#666',
              mb: 0.5
            }}>
              {exp.company || 'Kein Unternehmen angegeben'}
            </Typography>
            
            <Typography sx={{
              fontSize: '0.9rem',
              color: '#666',
              opacity: 0.8
            }}>
              {exp.duration || 'Kein Zeitraum angegeben'}
            </Typography>
          </Box>
        ))}
      </Box>

      {profile.summary && (
        <Box sx={{ mb: 4 }}>
          <Typography variant="h3" sx={{
            fontSize: '1.3rem',
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