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
      const months = days / 30.44;
      if (months < 12) return '#2e6f40';
      if (months < 24) return '#FFC03D';
      return '#d81b3b';
    } else {
      const percentage = value * 100;
      if (percentage < 40) return '#d81b3b';
      if (percentage >= 40 && percentage < 70) return '#FFC03D';
      return '#2e6f40';
    }
  };

  return (
    <Box sx={{ 
        bgcolor: '#fff', 
        borderRadius: '16px', 
        p: isMobile ? '20px' : '30px', 
        boxShadow: '0 2px 8px rgba(0,0,0,0.05)', 
        height: '95%', 
        display: 'flex', 
        flexDirection: 'column', 
        gap: 0, 
        overflow: 'hidden',
        position: 'relative',
        maxWidth: '100%',
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
      <Box sx={{ 
        display: 'flex', 
        alignItems: 'center', 
        gap: isMobile ? 1 : 2, 
        mb: 0, 
        minWidth: 0,
        height: isMobile ? 80 : 100, 
        flexShrink: 0
      }}>
        {candidate.imageUrl && candidate.imageUrl !== '' && (
          <img src={candidate.imageUrl} alt={name} style={{ width: isMobile ? 60 : 80, height: isMobile ? 60 : 80, borderRadius: '50%', objectFit: 'cover', border: '2px solid #eee', flexShrink: 0 }} />
        )}
        <Box sx={{ minWidth: 0 }}>
          <Typography variant="h3" sx={{ fontSize: isMobile ? '1rem' : '1.2rem', fontWeight: 600, color: '#1a1a1a', wordBreak: 'break-word', maxWidth: '100%', whiteSpace: 'normal' }}>{name}</Typography>
          <Link href={candidate.linkedinProfile} target="_blank" rel="noopener noreferrer" sx={{ fontSize: isMobile ? '0.7rem' : '0.8rem', color: '#001B41', textDecoration: 'none', '&:hover': { color: '#EB7836' }, wordBreak: 'break-all', maxWidth: '100%', display: 'block' }}>LinkedIn Profile</Link>
        </Box>
      </Box>
      <Box sx={{ 
        display: 'flex', 
        flexDirection: 'column', 
        flex: 1,
        justifyContent: 'space-between',
        mt: isMobile ? 1 : 2
      }}>
        <Box sx={{ 
          height: isMobile ? 50 : 60, 
          display: 'flex', 
          alignItems: 'center',
          mb: isMobile ? 1 : 1.5
        }}>
          {candidate.currentPosition ? (
            <Typography sx={{ color: '#666', fontSize: isMobile ? '0.85rem' : '1rem', wordBreak: 'break-word', maxWidth: '100%', whiteSpace: 'normal', lineHeight: 1.4 }}>
              <b>Current Position:</b> {candidate.currentPosition}
            </Typography>
          ) : (
            <Typography sx={{ color: '#ccc', fontSize: isMobile ? '0.85rem' : '1rem', fontStyle: 'italic' }}>
              <b>Current Position:</b> Not specified
            </Typography>
          )}
        </Box>

        {/* Location - feste Position */}
        <Box sx={{ 
          height: isMobile ? 50 : 60, 
          display: 'flex', 
          alignItems: 'center',
          mb: isMobile ? 1 : 1.5
        }}>
          {candidate.location ? (
            <Typography sx={{ color: '#666', fontSize: isMobile ? '0.85rem' : '1rem', wordBreak: 'break-word', maxWidth: '100%', whiteSpace: 'normal' }}>
              <b>Location:</b> {candidate.location}
            </Typography>
          ) : (
            <Typography sx={{ color: '#ccc', fontSize: isMobile ? '0.85rem' : '1rem', fontStyle: 'italic' }}>
              <b>Location:</b> Not specified
            </Typography>
          )}
        </Box>

        <Box sx={{ 
          height: isMobile ? 80 : 100, 
          display: 'flex', 
          flexDirection: 'column', 
          justifyContent: 'center'
        }}>
          {isTimeSeriesModel ? (
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: isMobile ? 0.5 : 1 }}>
              <Typography sx={{ color: '#666', fontSize: isMobile ? '0.85rem' : '1rem', fontWeight: 600 }}>Anticipated Change Date</Typography>
              <Typography sx={{ color: getConfidenceColor(confidence, true), fontSize: isMobile ? '0.95rem' : '1.1rem', fontWeight: 700 }}>
                {new Date(Date.now() + getDaysFromConfidence(confidence) * 24 * 60 * 60 * 1000).toLocaleDateString('en-US', { day: '2-digit', month: 'long', year: 'numeric' })}
              </Typography>
            </Box>
          ) : (
            <>
              <Typography sx={{ color: '#666', fontSize: isMobile ? '0.85rem' : '1rem', fontWeight: 600 }}>Job Change Confidence</Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: isMobile ? 1 : 1.5 }}>
                <Typography sx={{ fontWeight: 600, minWidth: isMobile ? 40 : 50, color: getConfidenceColor(confidence, false), fontSize: isMobile ? '0.8rem' : '0.95rem' }}>
                  {(confidence * 100).toFixed(0)}%
                </Typography>
                <Box sx={{ flexGrow: 1, height: isMobile ? 6 : 8, bgcolor: '#eee', borderRadius: 1, overflow: 'hidden' }}>
                  <Box sx={{ height: '100%', width: `${confidence * 100}%`, bgcolor: getConfidenceColor(confidence, false), borderRadius: 1, transition: 'width 0.3s ease' }} />
                </Box>
              </Box>
            </>
          )}
        </Box>
      </Box>
    </Box>
  );
};

export default CandidateCard; 