import React, { useState } from 'react';
import { Box, Paper } from '@mui/material';
import { Link, useLocation } from 'react-router-dom';

const Sidebar = () => {
  const location = useLocation();
  const [hovered, setHovered] = useState(false);

  const expanded = hovered;

  const menuItems = [
    { 
      path: '/', 
      icon: '/static/images/carrer_nav.jpeg',
      text: 'Manuelle-Prognose',
      alt: 'Karriere Navigation'
    },
    { 
      path: '/linkedin', 
      icon: '/static/images/linkedin_nav.jpg',
      text: 'LinkedIn-Prognose',
      alt: 'LinkedIn Navigation'
    },
    { 
      path: '/batch', 
      icon: '/static/images/candi_batch_nav.jpg',
      text: 'Batch-Upload',
      alt: 'Batch Navigation'
    },
    { 
      path: '/candidates', 
      icon: '/static/images/candidates.png',
      text: 'Kandidaten',
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
        width: expanded ? '240px' : '160px',
        height: '100vh',
        bgcolor: '#fff',
        borderRadius: 0,
        boxShadow: '0 8px 32px rgba(0,0,0,0.08)',
        transition: 'width 0.25s cubic-bezier(.4,0,.2,1)',
        zIndex: 1000,
        overflow: 'hidden',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'space-between',
        border: 'none',
      }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      <Box sx={{ width: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', pt: expanded ? 2.5 : 1.5, pb: expanded ? 2 : 1 }}>
        <img 
          src="/static/images/logo.png"
          alt="Aurio Technology Logo"
          style={{ width: expanded ? '120px' : '90px', height: 'auto', marginBottom: expanded ? 12 : 6, transition: 'width 0.2s, margin-bottom 0.2s' }}
        />
        <img 
          src="/static/images/ur-logo.png"
          alt="UR Logo"
          style={{  width: expanded ? '120px' : '90px',  height: 'auto', marginBottom: expanded ? 18 : 8, transition: 'width 0.2s, margin-bottom 0.2s' }}
        />
      </Box>
      {/* Navigation */}
      <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 2, alignItems: 'center', width: '100%', mt: 3 }}>
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
              borderRadius: 0,
              py: 2.5,
              px: 0,
              width: '100%',
              transition: 'background 0.2s',
              '&:hover': {
                bgcolor: 'rgba(255,128,0,0.13)',
              },
            }}
          >
            <img
              src={item.icon}
              alt={item.alt}
              style={{
                width: 38,
                height: 38,
                objectFit: 'contain',
                marginBottom: expanded ? 0 : 0,
                filter: 'grayscale(1) brightness(1.4)',
              }}
            />
            {expanded && (
              <Box
                sx={{
                  color: location.pathname === item.path ? '#FF8000' : '#222',
                  fontWeight: location.pathname === item.path ? 700 : 500,
                  fontSize: '0.8rem',
                  textAlign: 'center',
                  whiteSpace: 'nowrap',
                  letterSpacing: 0.2,
                  mt: 0.2,
                }}
              >
                {item.text}
              </Box>
            )}
          </Box>
        ))}
      </Box>
    </Paper>
  );
};

export default Sidebar; 