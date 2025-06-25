import React from 'react';
import { Box, Typography, Link, useTheme, useMediaQuery } from '@mui/material';
import IconButton from '@mui/material/IconButton';
import DeleteIcon from '@mui/icons-material/Delete';

const CandidateCard = ({ candidate, onDelete }) => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
  const name = `${candidate.firstName || ''} ${candidate.lastName || ''}`.trim();
  const isTimeSeriesModel = candidate.modelType === 'tft' || candidate.modelType === 'gru';
  const confidence = candidate.confidence ? candidate.confidence[0] : 0;

  const getDaysFromConfidence = (conf) => (!conf ? 0 : Math.round(conf));
  const getConfidenceColor = (value, isTimeSeries) => {
    if (isTimeSeries) {
      const days = getDaysFromConfidence(value);
      if (days <= 180) return '#8AD265';
      if (days <= 365) return '#FFC03D';
      return '#FF2525';
    } else {
      const percentage = value * 100;
      if (percentage < 40) return '#FF2525';
      if (percentage < 70) return '#FFC03D';
      return '#8AD265';
    }
  };

  return (
    <Box sx={{ 
        bgcolor: '#fff', 
        borderRadius: '16px', 
        p: isMobile ? '20px' : '30px', 
        boxShadow: { xs: 1, md: 2 }, 
        height: '95%', 
        display: 'flex', 
        flexDirection: 'column', 
        gap: isMobile ? 1 : 2, 
        overflow: 'hidden',
        position: 'relative',
        maxWidth: 480,
        minWidth: 320,
        width: '100%',
        '&:hover .delete-icon': {
            opacity: 1
        }
    }}>
        <IconButton
            className="delete-icon"
            onClick={(e) => {
                e.stopPropagation();
                onDelete(candidate._id);
            }}
            sx={{
                position: 'absolute',
                top: 8,
                right: 8,
                color: 'red',
                bgcolor: 'rgba(255, 255, 255, 0.7)',
                opacity: 0,
                transition: 'opacity 0.2s ease-in-out',
                '&:hover': {
                    bgcolor: 'rgba(255, 255, 255, 0.9)',
                }
            }}
        >
            <DeleteIcon />
        </IconButton>

      <Box sx={{ display: 'flex', alignItems: 'center', gap: isMobile ? 1 : 2, mb: isMobile ? 0.5 : 1, minWidth: 0 }}>
        {candidate.imageUrl && candidate.imageUrl !== '' && (
          <img src={candidate.imageUrl} alt={name} style={{ width: isMobile ? 60 : 80, height: isMobile ? 60 : 80, borderRadius: '50%', objectFit: 'cover', border: '2px solid #eee', flexShrink: 0 }} />
        )}
        <Box sx={{ minWidth: 0 }}>
          <Typography variant="h3" sx={{ fontSize: isMobile ? '1rem' : '1.2rem', fontWeight: 600, color: '#1a1a1a', wordBreak: 'break-word', maxWidth: '100%', whiteSpace: 'normal' }}>{name}</Typography>
          <Link href={candidate.linkedinProfile} target="_blank" rel="noopener noreferrer" sx={{ fontSize: isMobile ? '0.7rem' : '0.8rem', color: '#001B41', textDecoration: 'none', '&:hover': { color: '#EB7836' }, wordBreak: 'break-all', maxWidth: '100%', display: 'block' }}>LinkedIn Profile</Link>
        </Box>
      </Box>
      {candidate.currentPosition && (
        <Typography sx={{ color: '#666', fontSize: isMobile ? '0.85rem' : '1rem', mt: isMobile ? 0.5 : 1, wordBreak: 'break-word', maxWidth: '100%', whiteSpace: 'normal' }}><b>Current Position:</b> {candidate.currentPosition}</Typography>
      )}
      {candidate.location && (
        <Typography sx={{ color: '#666', fontSize: isMobile ? '0.85rem' : '1rem', wordBreak: 'break-word', maxWidth: '100%', whiteSpace: 'normal' }}><b>Location:</b> {candidate.location}</Typography>
      )}
      {candidate.industry && (
        <Typography sx={{ color: '#666', fontSize: isMobile ? '0.85rem' : '1rem', wordBreak: 'break-word', maxWidth: '100%', whiteSpace: 'normal' }}><b>Industry:</b> {candidate.industry}</Typography>
      )}
      <Box sx={{ mt: isMobile ? 1 : 2 }}>
        {isTimeSeriesModel ? (
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: isMobile ? 1 : 1.5, mb: isMobile ? 1 : 2 }}>
            <Typography sx={{ color: '#666', fontSize: isMobile ? '0.85rem' : '1rem', fontWeight: 600 }}>Anticipated Change Date</Typography>
            <Typography sx={{ color: getConfidenceColor(confidence, true), fontSize: isMobile ? '0.95rem' : '1.1rem', fontWeight: 700 }}>{new Date(Date.now() + getDaysFromConfidence(confidence) * 24 * 60 * 60 * 1000).toLocaleDateString('en-US', { day: '2-digit', month: 'long', year: 'numeric' })}</Typography>
          </Box>
        ) : (
          <>
            <Typography sx={{ color: getConfidenceColor(confidence, false), fontWeight: 600, mb: isMobile ? 0.5 : 1, fontSize: isMobile ? '0.85rem' : '1rem' }}>{confidence * 100 <= 60 ? 'Unlikely to consider a change' : confidence * 100 <= 80 ? 'Might consider a change' : 'Likely to consider a change'}</Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: isMobile ? 1 : 1.5, mb: isMobile ? 1 : 2 }}>
              <Typography sx={{ fontWeight: 600, minWidth: isMobile ? 40 : 50, color: getConfidenceColor(confidence, false), fontSize: isMobile ? '0.8rem' : '0.95rem' }}>{(confidence * 100).toFixed(0)}%</Typography>
              <Box sx={{ flexGrow: 1, height: isMobile ? 6 : 8, bgcolor: '#eee', borderRadius: 1, overflow: 'hidden' }}>
                <Box sx={{ height: '100%', width: `${confidence * 100}%`, bgcolor: getConfidenceColor(confidence, false), borderRadius: 1, transition: 'width 0.3s ease' }} />
              </Box>
            </Box>
          </>
        )}
      </Box>
    </Box>
  );
};

export default CandidateCard; 