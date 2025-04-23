import React from 'react';
import { Box, Typography, Link } from '@mui/material';

const CandidateCard = ({ candidate }) => {
  const name = `${candidate.firstName || ''} ${candidate.lastName || ''}`.trim();
  const confidence = candidate.confidence ? candidate.confidence[0] * 100 : 0;

  const getConfidenceColor = (confidence) => {
    if (confidence <= 50) return '#dc3545'; // rot
    if (confidence <= 75) return '#ffc107'; // gelb
    return '#28a745'; // grün
  };

  return (
    <Box sx={{
      bgcolor: '#fff',
      borderRadius: '16px',
      p: '30px',
      boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
      height: '100%',
      display: 'flex',
      flexDirection: 'column',
      gap: 2
    }}>
      <Typography variant="h3" sx={{
        fontSize: '1.2rem',
        fontWeight: 600,
        color: '#1a1a1a'
      }}>
        {name}
      </Typography>

      <Link 
        href={candidate.linkedinProfile} 
        target="_blank" 
        rel="noopener noreferrer"
        sx={{
          color: '#001B41',
          textDecoration: 'none',
          '&:hover': {
            color: '#FF5F00'
          }
        }}
      >
        LinkedIn Profil
      </Link>

      <Box sx={{ mt: 2 }}>
        <Typography sx={{
          color: getConfidenceColor(confidence),
          fontWeight: 600,
          mb: 1
        }}>
          {confidence <= 50 ? 'Geringe Wechselwahrscheinlichkeit' :
           confidence <= 75 ? 'Mittlere Wechselwahrscheinlichkeit' :
           'Hohe Wechselwahrscheinlichkeit'}
        </Typography>

        <Box sx={{ 
          display: 'flex',
          alignItems: 'center',
          gap: 1.5,
          mb: 2
        }}>
          <Typography sx={{ 
            fontWeight: 600, 
            minWidth: 50,
            color: getConfidenceColor(confidence)
          }}>
            {confidence.toFixed(0)}%
          </Typography>
          <Box sx={{ flexGrow: 1, height: 8, bgcolor: '#eee', borderRadius: 1, overflow: 'hidden' }}>
            <Box
              sx={{
                height: '100%',
                width: `${confidence}%`,
                bgcolor: getConfidenceColor(confidence),
                borderRadius: 1,
                transition: 'width 0.3s ease'
              }}
            />
          </Box>
        </Box>
      </Box>
    </Box>
  );
};

export default CandidateCard; 