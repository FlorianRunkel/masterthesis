import React, { useState, useEffect } from 'react';
import { Container, Typography, Box, Card, CardContent, Grid, Button, Paper, List, ListItem, ListItemIcon, ListItemText } from '@mui/material';
import { useNavigate } from 'react-router-dom';
import { useTheme } from '@mui/material/styles';
import { useMediaQuery } from '@mui/material';
import PersonIcon from '@mui/icons-material/Person';
import LinkedInIcon from '@mui/icons-material/LinkedIn';
import UploadFileIcon from '@mui/icons-material/UploadFile';
import DescriptionIcon from '@mui/icons-material/Description';
import CheckCircleOutlineIcon from '@mui/icons-material/CheckCircleOutline';
import ArrowForwardIcon from '@mui/icons-material/ArrowForward';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import TrackChangesIcon from '@mui/icons-material/TrackChanges';
import StorageIcon from '@mui/icons-material/Storage';
import AccountCircleIcon from '@mui/icons-material/AccountCircle';
import AccountBoxIcon from '@mui/icons-material/AccountBox';
import PeopleAltIcon from '@mui/icons-material/PeopleAlt';
import VerifiedIcon from '@mui/icons-material/Verified';
import BusinessCenterIcon from '@mui/icons-material/BusinessCenter';

const Index = () => {
  const navigate = useNavigate();
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const [user, setUser] = useState(null);

  useEffect(() => {
    const storedUser = localStorage.getItem('user');
    if (storedUser) {
      setUser(JSON.parse(storedUser));
    }
  }, []);

  const options = [
    {
      title: 'Manual Input',
      description: 'Enter the candidate data manually for a detailed career prediction.',
      icon: <DescriptionIcon sx={{ fontSize: 28, color: '#fff' }} />,
      path: '/manual',
      color: '#001242',
      bgColor: '#001242',
      features: ['Manual Data Entry', 'Instant Results', 'Personalized Recommendations']
    },
    {
      title: 'LinkedIn Profile',
      description: 'Analyze a LinkedIn profile by entering the profile URL.',
      icon: <LinkedInIcon sx={{ fontSize: 28, color: '#fff' }} />,
      path: '/linkedin',
      color: '#0077B5',
      bgColor: '#0077B5',
      features: ['Automatic Data Extraction', 'Social Validation', 'Network Analysis']
    },
    {
      title: 'Batch Upload',
      description: 'Upload multiple profiles at once for a mass analysis.',
      icon: <UploadFileIcon sx={{ fontSize: 28, color: '#fff' }} />,
      path: '/batch',
      color: '#EB7836',
      bgColor: '#EB7836',
      features: ['Mass Analysis', 'CSV Import', 'Comparison Analysis']
    }
  ];

  const benefits = [
    {
        icon: <AutoAwesomeIcon fontSize="large" sx={{color: '#002442'}} />,
        title: 'AI-supported Analysis',
        description: 'Modern Machine-Learning-Algorithms.'
    },
    {
        icon: <TrackChangesIcon fontSize="large" sx={{color: '#002442'}} />,
        title: 'Precise Predictions',
        description: '91% accuracy in job change predictions.'
    },
    {
        icon: <StorageIcon fontSize="large" sx={{color: '#002442'}} />,
        title: 'Comprehensive Database',
        description: 'Over 120,000 analyzed profiles.'
    },
    {
        icon: <AccountBoxIcon fontSize="large" sx={{color: '#001242'}} />,
        title: 'Candidate Management',
        description: 'Save promising candidates to contact them at the right time.'
    }
  ];

  const stats = [
    { 
      value: '120K+', 
      label: 'Analyzed Profiles', 
      icon: <PeopleAltIcon sx={{ fontSize: 38, color: '#002442' }} /> 
    },
    { 
      value: '91%', 
      label: 'Accuracy', 
      icon: <VerifiedIcon sx={{ fontSize: 38, color: '#002442' }} /> 
    },
    { 
      value: '24', 
      label: 'Industries', 
      icon: <BusinessCenterIcon sx={{ fontSize: 38, color: '#002442' }} /> 
    },
  ]

  return (
    <Box>
      {/* Hero Section */}
      <Box sx={{ mb: 6 }}>
        <Typography variant="h1" sx={{
          fontSize: { xs: '2.5rem', sm: '3.5rem', md: '4rem' },
          fontWeight: 800,
          color: '#002442',
          letterSpacing: '-1px'
        }}>
          Job Change Prediction AI
        </Typography>
        <Typography sx={{
          fontSize: { xs: '1rem', md: '1.25rem' },
          color: '#555',
          mt: 2,
          mb: 4,
          lineHeight: 1.6
        }}>
        Enhance your recruiting decisions with data-driven insights: This interactive dashboard visualizes the probability of a candidate's career change based on their individual career history. The predictions are generated using advanced machine learning models and made interpretable through Explainable AI techniques. Select one of the options below to begin the analysis.
        </Typography>

        <Grid
          container
          spacing={0}
          sx={{
            maxWidth: '900px',
            mt: 4,
            mx: 'auto',
            display: 'flex',
            flexDirection: 'row',
            justifyContent: 'center',
            alignItems: 'stretch',
          }}
        >
          {stats.map((stat, idx) => (
            <Grid
              item
              xs={4}
              key={stat.label}
              sx={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                py: { xs: 2, sm: 3 },
                px: { xs: 0.5, sm: 2 },
                borderRight: idx < stats.length - 1 ? '2px solid #e0e0e0' : 'none',
              }}
            >
              <Box sx={{ mb: 1 }}>
                {React.cloneElement(stat.icon, {
                  sx: {
                    fontSize: { xs: 32, sm: 38, md: 44 },
                    color: '#002442',
                  },
                })}
              </Box>
              <Typography
                variant="h2"
                sx={{
                  fontWeight: 900,
                  color: '#002442',
                  fontSize: { xs: '1.7rem', sm: '2.2rem', md: '2.8rem' },
                  mb: 0.5,
                  letterSpacing: '-1px',
                }}
              >
                {stat.value}
              </Typography>
              <Typography
                variant="subtitle1"
                sx={{
                  color: '#002442',
                  fontWeight: 600,
                  fontSize: { xs: '0.95rem', sm: '1.1rem' },
                  opacity: 0.85,
                }}
              >
                {stat.label}
              </Typography>
            </Grid>
          ))}
        </Grid>
      </Box>

      {/* Options Section */}
      <Box>
        <Grid container spacing={4}>
          {options.map((option) => (
            <Grid item xs={12} lg={4} key={option.title}>
              <Card 
                sx={{
                  height: '100%',
                  borderRadius: 4,
                  boxShadow: '0 16px 32px rgba(0, 0, 0, 0.05)',
                  transition: 'all 0.3s ease',
                  '&:hover': {
                    transform: 'translateY(-8px)',
                    boxShadow: `0 24px 48px rgba(0,0,0,0.1)`
                  },
                  display: 'flex',
                  flexDirection: 'column'
                }}
              >
                <CardContent sx={{ p: 4, flexGrow: 1, display: 'flex', flexDirection: 'column' }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', width: 64, height: 64, borderRadius: '50%', mb: 2, bgcolor: option.bgColor, justifyContent: 'center' }}>
                    {option.icon}
                  </Box>
                  <Typography variant="h5" component="h2" sx={{ fontWeight: 700, mb: 1.5, color: '#002442' }}>
                    {option.title}
                  </Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 1.5, minHeight: 40 }}>
                    {option.description}
                  </Typography>
                  <List dense sx={{ mb: 2, mt: 0, pt: 0 }}>
                    {option.features.map(feature => (
                      <ListItem key={feature} sx={{ p: 0, mb: 0.5 }}>
                        <ListItemIcon sx={{ minWidth: 32 }}>
                          <CheckCircleOutlineIcon sx={{ color: option.color, fontSize: 20 }} />
                        </ListItemIcon>
                        <ListItemText primary={feature} />
                      </ListItem>
                    ))}
                  </List>
                  <Box sx={{ mt: 'auto', pt: 1 }}>
                    <Button
                      fullWidth
                      variant="contained"
                      onClick={() => navigate(option.path)}
                      endIcon={<ArrowForwardIcon />}
                      sx={{
                        backgroundColor: option.color,
                        color: 'white',
                        fontWeight: 700,
                        borderRadius: '8px',
                        py: 1.5,
                        textTransform: 'none',
                        boxShadow: 'none',
                        fontSize: '1.05rem',
                        '&:hover': {
                          backgroundColor: option.color,
                          opacity: 0.9,
                          boxShadow: `0 4px 24px ${option.color}40`
                        }
                      }}
                    >
                      {option.title === 'Manual Input' && 'Manual Prediction'}
                      {option.title === 'LinkedIn Profile' && 'LinkedIn Prediction'}
                      {option.title === 'Batch Upload' && 'Batch Prediction'}
                    </Button>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      </Box>
      
      {/* Benefits Section */}
      <Box sx={{ py: { xs: 6, md: 10 } }}>
        <Typography variant="h2" sx={{ fontWeight: 700, color: '#002442', mb: 2, textAlign: 'center' }}>
            Why Career Prediction AI?
        </Typography>
        <Typography variant="h6" sx={{ fontWeight: 400, color: '#555', maxWidth: '700px', mb: 6, textAlign: 'center', mx: 'auto' }}>
            Use the power of artificial intelligence for job change predictions.
        </Typography>
        <Grid container spacing={{ xs: 3, md: 4 }}>
            {benefits.map((benefit, index) => (
                <Grid item xs={6} sm={6} md={3} key={index}>
                    <Paper elevation={0} sx={{ textAlign: 'center', p: { xs: 2, sm: 3 }, bgcolor: 'transparent' }}>
                        <Box sx={{color: '#F59E42', mb: 2}}>{benefit.icon}</Box>
                        <Typography variant="h6" sx={{ fontWeight: 700, color: '#222', mb: 1 }}>{benefit.title}</Typography>
                        <Typography variant="body2" color="text.secondary">{benefit.description}</Typography>
                    </Paper>
                </Grid>
            ))}
        </Grid>
      </Box>

      {/* Collaboration Section */}
      <Box sx={{ bgcolor: '#fff' }}>
        <Typography variant="h2" sx={{ textAlign: 'center', fontWeight: 700, color: '#002442', mb: 6 }}>
            Academic-Industry Collaboration
        </Typography>
        <Typography variant="h6" sx={{ fontWeight: 400,textAlign: 'center', color: '#555', mb: 6, mx: 'auto' }}>
            This application was developed as part of a Master's Thesis in Information Systems, combining academic research with practical AI application in partnership with aurio Technology GmbH and the University of Regensburg.
        </Typography>
        <Grid container spacing={4} justifyContent="center">
          {[
            { name: 'aurio Technology GmbH', logo: '/static/images/logo.png', url: 'https://www.aurio.ai/de/' },
            { name: 'University of Regensburg', logo: '/static/images/ur-logo.png', url: 'https://www.ur.de/' }
          ].map((partner) => (
            <Grid item xs={12} sm={8} md={5} key={partner.name}>
              <a href={partner.url} target="_blank" rel="noopener noreferrer" style={{ textDecoration: 'none' }}>
                <Paper 
                  elevation={2} 
                  sx={{ 
                    p: 4, 
                    textAlign: 'center', 
                    height: '100%',
                    display: 'flex',
                    flexDirection: 'column',
                    justifyContent: 'center',
                    alignItems: 'center',
                    borderRadius: 4,
                    transition: 'all 0.3s ease',
                    '&:hover': {
                      transform: 'translateY(-8px)',
                      boxShadow: '0 24px 48px rgba(0,0,0,0.1)'
                    }
                  }}
                >
                  <img src={partner.logo} alt={`${partner.name} Logo`} style={{ maxHeight: '60px', width: 'auto', marginBottom: '16px' }} />
                  <Typography variant="h6" sx={{ fontWeight: 600, color: 'text.primary' }}>
                    {partner.name}
                  </Typography>
                </Paper>
              </a>
            </Grid>
          ))}
        </Grid>
      </Box>
    </Box>
  );
};

export default Index; 