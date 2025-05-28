import React, { useState } from 'react';
import { Box, Paper } from '@mui/material';
import { Link, useLocation } from 'react-router-dom';

const Sidebar = () => {
  const location = useLocation();

  const menuItems = [
    { 
      path: '/', 
      icon: '/static/images/carrer_nav.jpeg',
      text: 'Manual Prediction', 
      alt: 'Karriere Navigation'
    },
    { 
      path: '/linkedin', 
      icon: '/static/images/linkedin_nav.jpg',
      text: 'LinkedIn Prediction',
      alt: 'LinkedIn Navigation'
    },
    { 
      path: '/batch', 
      icon: '/static/images/candi_batch_nav.jpg',
      text: 'Batch Upload',
      alt: 'Batch Navigation'
    },
    { 
      path: '/candidates', 
      icon: '/static/images/candidates.png',
      text: 'Candidates',
      alt: 'Kandidaten Navigation'
    }
  ];

  const bottomButtons = [
    { icon: '/static/images/settings.svg', alt: 'Einstellungen' },
    { icon: '/static/images/info.svg', alt: 'Info' },
    { icon: '/static/images/plus.svg', alt: 'Hinzufügen' },
    { icon: '/static/images/arrow.svg', alt: 'Weiter' },
  ];

  return (
    <Paper
      elevation={3}
      sx={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '260px',
        height: '100vh',
        bgcolor: '#fff',
        borderRadius: 0,
        boxShadow: '0 8px 32px rgba(0,0,0,0.08)',
        zIndex: 1000,
        overflow: 'hidden',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'space-between',
        border: 'none',
        gap: 2,
      }}
    >
      <Box sx={{ width: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', pt: 2.5, pb: 2 }}>
        <a 
          href="https://www.aurio.ai/de/"
          target="_blank" 
          rel="noopener noreferrer"
          style={{ textDecoration: 'none' }}
        >
          <img 
            src="/static/images/logo.png"
            alt="Aurio Technology Logo"
            style={{ width: '120px', height: 'auto', marginBottom: 12}}
          />
        </a>
        <a 
          href="https://www-mis.ur.de/master" 
          target="_blank" 
          rel="noopener noreferrer"
          style={{ textDecoration: 'none' }}
        >
          <img 
            src="/static/images/ur-logo.png"
            alt="UR Logo"
            style={{ width: '120px', height: 'auto', marginBottom: 18}}
          />
        </a>
      </Box>
      {/* Navigation */}
      <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 1, alignItems: 'center', width: '100%', mt: 2 }}>
        {menuItems.map((item) => (
          <Box
            key={item.path}
            component={Link}
            to={item.path}
            sx={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              textDecoration: 'none',
              bgcolor: location.pathname === item.path ? 'rgba(255,128,0,0.08)' : 'transparent',
              borderRadius: "12px",
              py: 2.5,
              px: 0,
              width: '100%',
              transition: 'background 1.2s',
              '&:hover': {
                bgcolor: 'rgba(255,128,0,0.13)',
              },
              mt: 0.5,
            }}
          >
            <img
              src={item.icon}
              alt={item.alt}
              style={{
                width: 35,
                height: 35,
                objectFit: 'contain',
                marginBottom: 0,
                filter: location.pathname === item.path ? 'none' : 'brightness(0.7) grayscale(0.2)',
                opacity: location.pathname === item.path ? 1 : 0.8,
              }}
            />   
            <Box
            sx={{
              color: location.pathname === item.path ? '#FF8000' : '#222',
              fontWeight: location.pathname === item.path ? 600 : 600,
              fontSize: '0.88rem',
              textAlign: 'center',
              whiteSpace: 'nowrap',
              letterSpacing: 0.2,
              mt: 0.5,
            }}
          >
            {item.text}
          </Box>
          </Box>
        ))}
      </Box>
    </Paper>
  );
};

export default Sidebar; 