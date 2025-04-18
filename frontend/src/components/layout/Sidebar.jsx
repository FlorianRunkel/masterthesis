import React from 'react';
import { Box, Paper } from '@mui/material';
import { Link, useLocation } from 'react-router-dom';

const Sidebar = () => {
  const location = useLocation();

  const menuItems = [
    { 
      path: '/', 
      icon: '/static/images/carrer_nav.jpeg',
      text: 'Karriere-Prognose',
      alt: 'Karriere Navigation'
    },
    { 
      path: '/batch', 
      icon: '/static/images/candi_batch_nav.jpg',
      text: 'Batch-Prognose',
      alt: 'Batch Navigation'
    },
    { 
      path: '/linkedin', 
      icon: '/static/images/linkedin_nav.jpg',
      text: 'LinkedIn-Prognose',
      alt: 'LinkedIn Navigation'
    }
  ];

  return (
    <Paper
      elevation={0}
      sx={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '280px',
        height: '100vh',
        bgcolor: 'background.paper',
        borderRight: '1px solid',
        borderColor: 'divider',
        borderRadius: 0,
        padding: '20px 0',
        zIndex: 1000,
      }}
    >
      <Box sx={{
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        overflowY: 'auto'
      }}>
        {/* Logo Container */}
        <Box sx={{
          padding: '20px',
          marginBottom: '20px',
          display: 'flex',
          justifyContent: 'center'
        }}>
          <Box sx={{
            display: 'flex',
            flexDirection: 'column',
            gap: '20px',
            alignItems: 'center'
          }}>
            <img 
              src="/static/images/logo.png"
              alt="Aurio Technology Logo"
              style={{
                width: '120px',
                height: 'auto'
              }}
            />
            <img 
              src="/static/images/ur-logo.png"
              alt="UR Logo"
              style={{
                width: '120px',
                height: 'auto'
              }}
            />
          </Box>
        </Box>

        {/* Navigation Menu */}
        <Box sx={{
          display: 'flex',
          flexDirection: 'column',
          gap: '30px',
          padding: '0 20px'
        }}>
          {menuItems.map((item) => (
            <Box
              key={item.path}
              component={Link}
              to={item.path}
              sx={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                gap: '10px',
                padding: '15px',
                borderRadius: '8px',
                textDecoration: 'none',
                transition: 'background-color 0.3s ease',
                '&:hover': {
                  bgcolor: 'rgba(242, 242, 242, 0.6)',
                  '& .nav-text': {
                    color: '#001B41',
                  },
                },
              }}
            >
              <img
                src={item.icon}
                alt={item.alt}
                style={{
                  width: '48px',
                  height: '48px',
                  borderRadius: '8px',
                  objectFit: 'cover'
                }}
              />
              <Box
                className="nav-text"
                sx={{
                  color: location.pathname === item.path ? '#FF5F00' : '#333',
                  fontWeight: location.pathname === item.path ? 600 : 500,
                  fontSize: '14px',
                  textAlign: 'center',
                  width: '100%'
                }}
              >
                {item.text}
              </Box>
            </Box>
          ))}
        </Box>
      </Box>
    </Paper>
  );
};

export default Sidebar; 