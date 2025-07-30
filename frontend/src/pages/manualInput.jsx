import React, { useState, useRef, useEffect } from 'react';
import { Box, Typography, Button, TextField, Select, MenuItem, Checkbox, FormControlLabel, IconButton, Tooltip, Dialog, DialogTitle, DialogContent, DialogActions, Divider } from '@mui/material';
import PredictionResultTime from '../components/prediction/prediction_time';
import PredictionResultClassification from '../components/prediction/prediction_classification';
import DeleteOutlineIcon from '@mui/icons-material/DeleteOutline';
import { useTheme } from '@mui/material/styles';
import { useMediaQuery } from '@mui/material';
import SchoolIcon from '@mui/icons-material/School';
import BusinessCenterIcon from '@mui/icons-material/BusinessCenter';
import AddIcon from '@mui/icons-material/Add';
import Paper from '@mui/material/Paper';
import { API_BASE_URL } from '../api';
import HelpOutlineIcon from '@mui/icons-material/HelpOutline';
import InfoIcon from '@mui/icons-material/Info';
import WorkIcon from '@mui/icons-material/Work';
import LoadingSpinner from '../components/shared/loading_spinner';
import DescriptionIcon from '@mui/icons-material/Description';

const modelOptions = [
  {
    value: 'gru',
    title: 'Gated Recurrent Unit (GRU)',
    description: 'Sequence model for time series and career paths'
  },
  {
    value: 'xgboost',
    title: 'Extreme Gradient Boosting (XGBoost)',
    description: 'Powerful machine learning model for structured data'
  },
  {
    value: 'tft',
    title: 'Temporal Fusion Transformer (TFT)',
    description: 'Modern deep learning model for complex time series'
  }
];

const degreeOptions = [
  { value: 'phd', label: 'PhD' },
  { value: 'master', label: 'Master' },
  { value: 'bachelor', label: 'Bachelor' },
  { value: 'apprenticeship', label: 'Apprenticeship' },
  { value: 'other', label: 'Other' }
];

const ManualInput = () => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const isDesktop = useMediaQuery(theme.breakpoints.up('lg'));
  const [experiences, setExperiences] = useState(() => {
    const saved = localStorage.getItem('manualInputExperiences');
    return saved ? JSON.parse(saved) : [{
      company: '',
      position: '',
      startDate: '',
      endDate: ''
    }];
  });
  const [selectedModel, setSelectedModel] = useState(() => {
    const saved = localStorage.getItem('manualInputSelectedModel');
    return saved || '';
  });
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [education, setEducation] = useState(() => {
    const saved = localStorage.getItem('manualInputEducation');
    return saved ? JSON.parse(saved) : [{
      school: '',
      degree: '',
      fieldOfStudy: '',
      startDate: '',
      endDate: ''
    }];
  });
  const [showModelChangeHint, setShowModelChangeHint] = useState(false);
  const [predictionModelType, setPredictionModelType] = useState('');
  const predictionRef = useRef(null);
  const [showModelInfo, setShowModelInfo] = useState(false);

  const firstExperience = experiences[0];
  const secondExperience = experiences[1];
  
  const hasValidExperiences = firstExperience && 
    firstExperience.company && 
    firstExperience.position && 
    firstExperience.startDate && 
    (firstExperience.endDate || firstExperience.endDate === 'Present');

  const hasValidTFTExperiences = selectedModel === 'tft' ? 
    (hasValidExperiences && secondExperience && 
     secondExperience.company && 
     secondExperience.position && 
     secondExperience.startDate && 
     (secondExperience.endDate || secondExperience.endDate === 'Present')) : 
    hasValidExperiences;

  useEffect(() => {
    if (prediction && predictionRef.current) {
      predictionRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [prediction]);

  useEffect(() => {
    localStorage.setItem('manualInputExperiences', JSON.stringify(experiences));
  }, [experiences]);

  useEffect(() => {
    localStorage.setItem('manualInputEducation', JSON.stringify(education));
  }, [education]);

  useEffect(() => {
    localStorage.setItem('manualInputSelectedModel', selectedModel);
  }, [selectedModel]);

  useEffect(() => {
    if (prediction) {
      localStorage.setItem('manualPrediction', JSON.stringify(prediction));
      localStorage.setItem('manualPredictionModelType', predictionModelType);
    }
  }, [prediction, predictionModelType]);

  useEffect(() => {
    const saved = localStorage.getItem('manualPrediction');
    const savedType = localStorage.getItem('manualPredictionModelType');
    if (saved) {
      setPrediction(JSON.parse(saved));
      setPredictionModelType(savedType || '');
    }
  }, []);

  const handleAddExperience = () => {
    setExperiences([...experiences, {
      company: '',
      position: '',
      startDate: '',
      endDate: ''
    }]);
  };

  const handleRemoveExperience = (index) => {
    if (selectedModel === 'tft' && index === 1) {
      return;
    }
    
    if (experiences.length > 1) {
      const newExperiences = experiences.filter((_, i) => i !== index);
      setExperiences(newExperiences);
    }
  };

  const handleExperienceChange = (index, field, value) => {
    const newExperiences = [...experiences];
    newExperiences[index] = {
      ...newExperiences[index],
      [field]: value
    };
    setExperiences(newExperiences);
  };

  const handleAddEducation = () => {
    setEducation([...education, {
      school: '',
      degree: '',
      fieldOfStudy: '',
      startDate: '',
      endDate: ''
    }]);
  };

  const handleEducationChange = (index, field, value) => {
    const newEducation = [...education];
    newEducation[index] = {
      ...newEducation[index],
      [field]: value
    };
    setEducation(newEducation);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setPrediction(null);
    setLoading(true);
    setError(null);
    setShowModelChangeHint(false);
    setPredictionModelType(selectedModel);

    try {
      const filteredExperiences = experiences.filter(exp => 
        exp.company || exp.position || exp.startDate || exp.endDate
      );

      const filteredEducation = education.filter(edu =>
        edu.school || edu.degree || edu.fieldOfStudy || edu.startDate || edu.endDate
      );

      const formatDate = (dateStr, model) => {
        if (!dateStr || dateStr === 'Present') return dateStr;
        if (dateStr.includes('-')) {
            const [year, month, day] = dateStr.split('-');

            if (model === 'tft') {
                return `${day}/${month}/${year}`;
            } else {
                return `${month}/${year}`;
            }
        }
        return dateStr;
      };

      const profile_data = {
        firstName: "Unbekannt",
        lastName: "Unbekannt",
        linkedinProfileInformation: JSON.stringify({
          firstName: "Unbekannt",
          lastName: "Unbekannt",
          workExperience: filteredExperiences.map(exp => ({
            company: exp.company || "",
            position: exp.position || "",
            startDate: exp.startDate ? formatDate(exp.startDate, selectedModel) : "",
            endDate: exp.endDate === 'Present' ? 'Present' : (exp.endDate ? formatDate(exp.endDate, selectedModel) : ""),
            type: "fullTime",
            location: "",
            description: ""
          })),
          education: filteredEducation.map(edu => ({
            school: edu.school || "",
            degree: edu.degree || "",
            fieldOfStudy: edu.fieldOfStudy || "",
            startDate: edu.startDate ? formatDate(edu.startDate, selectedModel) : "",
            endDate: edu.endDate === 'Present' ? 'Present' : (edu.endDate ? formatDate(edu.endDate, selectedModel) : "")
          })),
          skills: [],
          location: "",
          headline: "",
          languageSkills: {}
        }),
        modelType: selectedModel.toLowerCase()
      };

      console.log("Sende Daten:", profile_data);

      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(profile_data),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Fehler bei der Vorhersage');
      }

      const data = await response.json();
      setPrediction(data);
    } catch (err) {
      console.error("Fehler:", err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleModelChange = (value) => {
    setSelectedModel(value);
    setShowModelChangeHint(true);
    
    if (value === 'tft' && experiences.length < 2) {
      const additionalExperiences = 2 - experiences.length;
      for (let i = 0; i < additionalExperiences; i++) {
        setExperiences(prev => [...prev, {
          company: '',
          position: '',
          startDate: '',
          endDate: ''
        }]);
      }
    }
    
    if (value !== 'tft' && experiences.length > 1) {
      const lastExperience = experiences[experiences.length - 1];
      const isLastExperienceEmpty = !lastExperience.company && !lastExperience.position && 
                                  !lastExperience.startDate && !lastExperience.endDate;
      
      if (isLastExperienceEmpty) {
        setExperiences(prev => prev.slice(0, -1));
      }
    }
  };
  return (
    <Box>

      <Typography variant="h1" sx={{ 
        fontSize: isMobile ? '1.8rem' : '2.5rem', 
        fontWeight: 700, 
        color: '#001242', 
        mb: 2 
      }}>
        Manual Prediction
      </Typography>
      <Typography sx={{ 
        color: '#666', 
        mb: 4, 
        fontSize: isMobile ? '0.9rem' : '1rem', 
        maxWidth: '800px' 
      }}>
        Analyze the job change probability of a single candidate based on their current work experience and educational background.
      </Typography>

      <Box component="form" onSubmit={handleSubmit} sx={{ width: '100%' }}>
        <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 4 }}>
          <Box sx={{ flex: 1 }}>
          <Paper sx={{ p: { xs: 2, sm: 3 }, mb: 4, borderRadius: '16px', boxShadow: '0 2px 8px rgba(0,0,0,0.05)' }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: education.length > 0 ? 3 : 0 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
                    <SchoolIcon sx={{ color: '#001242' }} />
                    <Typography variant="h6" sx={{ fontWeight: 600, color: '#001242' }}>Educational Background</Typography>
                </Box>
                <Button
                    variant="outlined"
                    startIcon={<AddIcon />}
                    onClick={handleAddEducation}
                    sx={{ textTransform: 'none', borderRadius: '8px', fontWeight: 600 , color: '#001242', borderColor: '#001242' }}
                >
                    Add
                </Button>
            </Box>
            {education.map((edu, index) => (
              <Box
                key={index}
                sx={{ 
                  mt: 2,
                  mb: 2, 
                  borderRadius: '12px', 
                  p: { xs: 1.5, sm: 2 }, 
                  border: '1px solid #e0e0e0', 
                  position: 'relative' 
                }}
              >
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1.6 }}>
                  <Typography sx={{ fontWeight: 700, color: '#001242', fontSize: '1.1rem' }}>
                    {index === 0 ? 'First Degree' : 'Further Studies'}  
                  </Typography>
                  {index > 0 && (
                    <Button
                      onClick={() => {
                        const newEducation = education.filter((_, i) => i !== index);
                        setEducation(newEducation);
                      }}
                      sx={{ color: '#FF2525', fontWeight: 600, fontSize: '0.8rem', textTransform: 'none', display: 'flex', alignItems: 'center', gap: 0.4, p: 0, minWidth: 0 }}
                      startIcon={<DeleteOutlineIcon />}
                    >
                      Remove
                    </Button>
                  )}
                </Box>
                <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr' }, gap: 2.4, mb: 1.6 }}>
                  <Box>
                    <Typography sx={{ fontSize: '0.95rem', fontWeight: 600, mb: 0.5 }}>School/University</Typography>
                    <TextField
                      value={edu.school}
                      size="small"
                      onChange={(e) => handleEducationChange(index, 'school', e.target.value)}
                      fullWidth
                      sx={{ '& .MuiOutlinedInput-root': { fontSize: '0.88rem', height: '46px' }, input: { fontSize: '0.88rem' } }}
                    />
                  </Box>
                  <Box>
                    <Typography sx={{ fontSize: '0.95rem', fontWeight: 600, mb: 0.5 }}>Degree</Typography>
                    <Select
                      displayEmpty
                      value={edu.degree}
                      size="small"
                      onChange={(e) => handleEducationChange(index, 'degree', e.target.value)}
                      fullWidth
                      sx={{ height: '46px', fontSize: '0.88rem' }}
                      renderValue={selected => selected ? degreeOptions.find(opt => opt.value === selected)?.label : <Typography sx={{color: 'text.secondary'}}>Select Degree</Typography>}
                    >
                      <MenuItem value="" disabled>
                        Degree
                      </MenuItem>
                      {degreeOptions.map((option) => (
                        <MenuItem value={option.value} key={option.value}>
                          {option.label}
                        </MenuItem>
                      ))}
                    </Select>
                  </Box>
                </Box>
                <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr' }, gap: 2.4, mb: 1.6 }}>
                  <Box sx={{ gridColumn: '1 / -1' }}>
                    <Typography sx={{ fontSize: '0.95rem', fontWeight: 600, mb: 0.5 }}>Field of Study</Typography>
                    <TextField
                      value={edu.fieldOfStudy}
                      size="small"
                      onChange={(e) => handleEducationChange(index, 'fieldOfStudy', e.target.value)}
                      fullWidth
                      sx={{ '& .MuiOutlinedInput-root': { fontSize: '0.88rem', height: '46px' }, input: { fontSize: '0.88rem' } }}
                    />
                  </Box>
                </Box>
                <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr' }, gap: 2.4, alignItems: 'flex-end' }}>
                  <Box>
                    <Typography sx={{ fontSize: '0.95rem', fontWeight: 600, mb: 0.5 }}>Start Date</Typography>
                    <TextField
                      type="date"
                      value={edu.startDate}
                      size="small"
                      onChange={(e) => handleEducationChange(index, 'startDate', e.target.value)}
                      InputLabelProps={{ shrink: true }}
                      fullWidth
                      sx={{ '& .MuiOutlinedInput-root': { fontSize: '0.88rem', height: '46px' }, input: { fontSize: '0.88rem' } }}
                    />
                  </Box>
                  <Box sx={{ display: 'flex', alignItems: 'flex-end', gap: 1 }}>
                    <Box sx={{ flex: 1 }}>
                      <Typography sx={{ fontSize: '0.95rem', fontWeight: 600, mb: 0.5 }}>End Date</Typography>
                      <TextField
                        type="date"
                        disabled={edu.endDate === 'Present'}
                        value={edu.endDate === 'Present' ? '' : edu.endDate}
                        size="small"
                        onChange={(e) => handleEducationChange(index, 'endDate', e.target.value)}
                        InputLabelProps={{ shrink: true }}
                        fullWidth
                        sx={{ '& .MuiOutlinedInput-root': { fontSize: '0.88rem', height: '46px' }, input: { fontSize: '0.88rem' } }}
                      />
                    </Box>
                    <FormControlLabel
                        control={
                            <Checkbox
                                checked={edu.endDate === 'Present'}
                                onChange={e => handleEducationChange(index, 'endDate', e.target.checked ? 'Present' : '')}
                            />
                        }
                        label="Present"
                        sx={{ mb: '2px', '& .MuiTypography-root': { fontSize: '0.88rem', fontWeight: 500 } }}
                    />
                  </Box>
                </Box>
              </Box>
            ))}
          </Paper> 
          <Paper sx={{ p: { xs: 2, sm: 3 }, mb: 4, borderRadius: '16px', boxShadow: '0 2px 8px rgba(0,0,0,0.05)' }}> 
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: experiences.length > 0 ? 3 : 0 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
                      <BusinessCenterIcon sx={{ color: '#001242' }} />
                      <Typography variant="h6" sx={{ fontWeight: 600, color: '#001242' }}>Work Experience</Typography>
                  </Box>
                  <Button
                      variant="outlined"
                      startIcon={<AddIcon />}
                      onClick={handleAddExperience}
                      sx={{ textTransform: 'none', borderRadius: '8px', fontWeight: 600 , color: '#001242', borderColor: '#001242'}}
                  >
                      Add
                  </Button>
              </Box>
              {experiences.map((exp, index) => (
                <Box
                  key={index}
                  sx={{ 
                    mt: 2,
                    mb: 2,  
                    borderRadius: '12px', 
                    p: { xs: 1.5, sm: 2 }, 
                    border: '1px solid #e0e0e0', 
                    position: 'relative'
                  }}
                >
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1.6 }}>
                    <Typography sx={{ fontWeight: 700, color: '#001242', fontSize: '1.1rem' }}>
                      {index === 0 ? 'Career Start' : 'Following Role'}
                    </Typography>
                    {index > 0 && !(selectedModel === 'tft' && index === 1) && (
                      <Button
                        onClick={() => handleRemoveExperience(index)}
                        sx={{ color: '#FF2525', fontWeight: 600, fontSize: '0.8rem', textTransform: 'none', display: 'flex',alignItems: 'center',  gap: 0.4,  p: 0, minWidth: 0 }} startIcon={<DeleteOutlineIcon />} >
                        Remove
                      </Button>
                    )}
                  </Box>
                  <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr' }, gap: 2.4, mb: 1.6 }}>
                    <Box>
                      <Typography sx={{ fontSize: '0.95rem', fontWeight: 600, mb: 0.5 }}>Company</Typography>
                      <TextField
                        value={exp.company}
                        size="small"
                        onChange={e => handleExperienceChange(index, 'company', e.target.value)}
                        fullWidth
                        sx={{ '& .MuiOutlinedInput-root': { fontSize: '0.88rem', height: '46px'}, input: { fontSize: '0.88rem'} }}
                      />
                    </Box>
                    <Box>
                      <Typography sx={{ fontSize: '0.95rem', fontWeight: 600, mb: 0.5 }}>Position</Typography>
                      <TextField
                        value={exp.position}
                        size="small"
                        onChange={e => handleExperienceChange(index, 'position', e.target.value)}
                        fullWidth
                        sx={{  '& .MuiOutlinedInput-root': { fontSize: '0.88rem', height: '46px' }, input: { fontSize: '0.88rem' } }}
                      />
                    </Box>
                  </Box>
                  <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr' }, gap: 2.4, alignItems: 'flex-end' }}>
                    <Box>
                      <Typography sx={{ fontSize: '0.95rem', fontWeight: 600, mb: 0.5 }}>Start Date</Typography>
                      <TextField
                        type="date"
                        value={exp.startDate}
                        size="small"
                        onChange={e => handleExperienceChange(index, 'startDate', e.target.value)}
                        InputLabelProps={{ shrink: true }}
                        fullWidth
                        sx={{  '& .MuiOutlinedInput-root': { fontSize: '0.88rem', height: '46px' }, input: { fontSize: '0.88rem' } }}
                      />
                    </Box>
                    <Box sx={{ display: 'flex', alignItems: 'flex-end', gap: 1 }}>
                      <Box sx={{ flex: 1 }}>
                        <Typography sx={{ fontSize: '0.95rem', fontWeight: 600, mb: 0.5 }}>End Date</Typography>
                        <TextField
                          size="small"
                          type="date"
                          disabled={exp.endDate === 'Present'}
                          value={exp.endDate === 'Present' ? '' : exp.endDate}
                          onChange={e => handleExperienceChange(index, 'endDate', e.target.value)}
                          InputLabelProps={{ shrink: true }}
                          fullWidth
                          sx={{  '& .MuiOutlinedInput-root': { fontSize: '0.88rem', height: '46px' }, input: { fontSize: '0.88rem' } }}
                        />
                      </Box>
                      <FormControlLabel
                          control={
                              <Checkbox
                                  checked={exp.endDate === 'Present'}
                                  onChange={e => handleExperienceChange(index, 'endDate', e.target.checked ? 'Present' : '')}
                              />
                          }
                          label="Present"
                          sx={{ mb: '2px', '& .MuiTypography-root': { fontSize: '0.88rem', fontWeight: 500 } }}
                      />
                    </Box>
                  </Box>
                </Box>
              ))}
          </Paper>
        </Box>
        {isDesktop && (

        <Paper sx={{ p: { xs: 2, sm: 3 }, mb: 4, borderRadius: '16px', boxShadow: '0 2px 8px rgba(0,0,0,0.05)', minWidth: 300, maxWidth: 550, height: 'fit-content', position: 'sticky', top: 20 }}>  
            <Box sx={{ 
              display: 'flex',
              alignItems: 'center',
              gap: 2,
            }}>
              <Box sx={{
                width: 48,
                height: 48,
                borderRadius: '16px',
                bgcolor: 'linear-gradient(135deg, #001242 0%, #1a365d 100%)',
                background: 'linear-gradient(135deg, #001242 0%, #1a365d 100%)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: 'white',
                fontSize: '1.4rem',
              }}>
                <DescriptionIcon />
              </Box>
              <Box>
                <Typography variant="h6" sx={{ fontWeight: 700, color: '#001242', fontSize: '1.1rem', mb: 0.5 }}>
                  Curriculum Vitae Overview
                </Typography>
                <Typography sx={{ color: '#666', fontSize: '0.8rem', fontWeight: 500, mb: 0.5 }}>
                  A summary of the candidate's resume
                </Typography>
              </Box>
            </Box>
            <Divider sx={{ my: 2, borderColor: '#e0e0e0', borderWidth: '1px', borderStyle: 'solid' }} />
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
              {education.filter(edu => edu.school || edu.degree || edu.fieldOfStudy).map((edu, idx) => (
                <Box key={"edu-"+idx} sx={{ display: 'flex', alignItems: 'flex-start', gap: 3 }}>
                  <Box sx={{ 
                    display: 'flex', 
                    flexDirection: 'column', 
                    alignItems: 'center',
                    position: 'relative'
                  }}>
                    <Box sx={{
                      width: 52,
                      height: 52,
                      borderRadius: '16px',
                      background: 'linear-gradient(135deg, #001242 0%, #1a365d 100%)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      color: 'white',
                      fontSize: '1.4rem',
                    }}>
                      <SchoolIcon />
                    </Box>
                    {idx < education.filter(edu => edu.school || edu.degree || edu.fieldOfStudy).length - 1 && (
                      <Box sx={{
                        width: 3,
                        height: 40,
                        background: 'linear-gradient(180deg, #001242 0%, #4a90e2 100%)',
                        borderRadius: '2px',
                        mt: 1,
                        opacity: 0.6
                      }} />
                    )}
                  </Box>
                  
                  <Box sx={{ flex: 1, pt: 0.5 }}>
                    <Typography sx={{ fontWeight: 700, color: '#001242', fontSize: '1rem', mb: 0.5 }}>
                      {edu.degree || 'Degree'}
                    </Typography>
                    <Typography sx={{ color: '#666', fontSize: '0.9rem', mb: 0.5, fontWeight: 500 }}>
                      {edu.school || 'School'}
                    </Typography>
                    {edu.fieldOfStudy && (
                      <Typography sx={{ color: '#888', fontSize: '0.85rem', mb: 0.5, fontStyle: 'italic' }}>
                        {edu.fieldOfStudy}
                      </Typography>
                    )}
                    <Typography sx={{ color: '#999', fontSize: '0.8rem', fontWeight: 500 }}>
                      {edu.startDate && edu.endDate ? `${edu.startDate} - ${edu.endDate === 'Present' ? 'Present' : edu.endDate}` : ''}
                    </Typography>
                  </Box>
                </Box>
              ))}
              
              {experiences.filter(exp => exp.company || exp.position).map((exp, idx) => (
                <Box key={"exp-"+idx} sx={{ display: 'flex', alignItems: 'flex-start', gap: 3 }}>
                  <Box sx={{ 
                    display: 'flex', 
                    flexDirection: 'column', 
                    alignItems: 'center',
                    position: 'relative'
                  }}>
                    <Box sx={{
                      width: 52,
                      height: 52,
                      borderRadius: '16px',
                      background: 'linear-gradient(135deg, #EB7836 0%, #f39c12 100%)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      color: 'white',
                      fontSize: '1.4rem',
                    }}>
                      <WorkIcon />
                    </Box>
                    {idx < experiences.filter(exp => exp.company || exp.position).length - 1 && (
                      <Box sx={{
                        width: 3,
                        height: 40,
                        background: 'linear-gradient(180deg, #EB7836 0%, #f39c12 100%)',
                        borderRadius: '2px',
                        mt: 1,
                        opacity: 0.6
                      }} />
                    )}
                  </Box>
                  
                  <Box sx={{ flex: 1, pt: 0.5 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mb: 0.5 }}>
                      <Typography sx={{ fontWeight: 700, color: '#001242', fontSize: '1rem' }}>
                        {exp.position || 'Position'}
                      </Typography>
                      {exp.endDate === 'Present' && (
                        <Box sx={{ 
                          background: 'linear-gradient(135deg, #4CAF50 0%, #66bb6a 100%)',
                          color: 'white', 
                          px: 1.5, 
                          py: 0.3, 
                          borderRadius: '20px', 
                          fontSize: '0.75rem',
                          fontWeight: 600,
                        }}>
                          Current
                        </Box>
                      )}
                    </Box>
                    <Typography sx={{ color: '#666', fontSize: '0.9rem', mb: 0.5, fontWeight: 500 }}>
                      {exp.company || 'Company'}
                    </Typography>
                    <Typography sx={{ color: '#999', fontSize: '0.8rem', fontWeight: 500 }}>
                      {exp.startDate && exp.endDate ? `${exp.startDate} - ${exp.endDate === 'Present' ? 'Present' : exp.endDate}` : ''}
                    </Typography>
                  </Box>
                </Box>
              ))}
              
              {education.filter(edu => edu.school || edu.degree || edu.fieldOfStudy).length === 0 && experiences.filter(exp => exp.company || exp.position).length === 0 && (
                <Box sx={{ 
                  textAlign: 'center', 
                  py: 4,
                  color: '#999'
                }}>
                  <SchoolIcon sx={{ fontSize: 48, mb: 2, opacity: 0.3 }} />
                  <Typography sx={{ fontStyle: 'italic', fontSize: '0.9rem', fontWeight: 500 }}>
                      Add work experience and education to see the timeline.
                  </Typography>
                </Box>
              )}
            </Box>
          </Paper>
        )}
      </Box>
      <Box sx={{ mt: 4, bgcolor: '#fff', borderRadius: '16px', p: 4, boxShadow: '0 2px 8px rgba(0,0,0,0.08)'}}>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 0.8 }}>
          <Typography variant="h2" sx={{ fontSize: '1.36rem', fontWeight: 700, color: '#001242' }}>
            Select AI model
          </Typography>
          <Tooltip 
            title="Click to learn more about each model"
            placement="top"
            arrow
          >
            <IconButton 
              size="small"
              onClick={() => setShowModelInfo(true)}
              sx={{ 
                color: '#001242',
                '&:hover': { bgcolor: '#f5f5f5' }
              }}
            >
              <HelpOutlineIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        </Box>
        <Typography sx={{ color: '#888', mb: 3.2, fontSize: '0.86rem' }}>
          Select the appropriate model for a precise prediction.
        </Typography>
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.6, mb: 1.6 }}>
          {modelOptions.map(option => (
            <Box key={option.value} onClick={() => handleModelChange(option.value)} sx={{cursor: 'pointer',bgcolor: '#fff', border: selectedModel === option.value ? '2px solid #EB7836' : '1.5px solid #e3e6f0', borderRadius: '16px', p: 3, boxShadow: selectedModel === option.value ? '0 2px 8px rgba(59,71,250,0.08)' : 'none', transition: 'all 0.2s', display: 'flex', flexDirection: 'column', outline: selectedModel === option.value ? '2px solid #EB7836' : 'none'}} >
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 0.4 }}>
                <Typography sx={{ fontWeight: 700, fontSize: '0.94rem', color: '#1a1a1a' }}>
                  {option.title}
                </Typography>
                {option.value === 'tft' && (
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.3 }}>
                    <InfoIcon sx={{ color: '#EB7836', fontSize: '0.8rem' }} />
                    <Typography sx={{ color: '#EB7836', fontSize: '0.7rem', fontWeight: 700, letterSpacing: 0.3}}>
                      Requires at least 2 Work Experience entries
                    </Typography>
                  </Box>
                )}
              </Box>
              <Typography sx={{ color: '#888', fontSize: '0.84rem' }}>
                {option.description}
              </Typography>
            </Box>
          ))}
        </Box>
        {showModelChangeHint && (
          <Box sx={{ bgcolor: '#FFF8E1', border: '1px solid #FFD54F', color: '#EB7836', p: 2, borderRadius: 2, mb: 1, fontSize: '0.8rem'}}>
            Please click 'Start prediction' to run the new model.
          </Box>
        )}
        <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 2 }}>
          <Button type="submit" disabled={loading || !selectedModel || !hasValidTFTExperiences} sx={{
            minWidth: 256,
            px: 3.2,
            py: 1.44,
            fontSize: '0.94rem',
            fontWeight: 700,
            borderRadius: '11.2px',
            color: '#fff',
            background: 'linear-gradient(90deg, #EB7836 0%, #EB7836 100%)',
            boxShadow: '0 2px 8px rgba(108,99,255,0.05)',
            textTransform: 'none',
            letterSpacing: 0.16,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 1.2,
            mt: 1.6,
            mb: 6,
            '&:hover': {
              background: 'linear-gradient(90deg, #EB7836 0%, #EB7836 100%)',
            },
            '&.Mui-disabled': {
              background: '#e3e6f0',
              color: '#bdbdbd',
            },
          }}>
            Start prediction
          </Button>
        </Box>
        <Dialog
          open={showModelInfo}
          onClose={() => setShowModelInfo(false)}
          fullWidth
          maxWidth="sm"
          PaperProps={{
            sx: {
              borderRadius: 3,
              maxWidth: { xs: '95vw', sm: 600, md: 800 },
              maxHeight: { xs: '95vh', sm: '95vh', md: '95vh' },
            }
          }}
        >
          <DialogTitle
            sx={{
              fontSize: { xs: '1.1rem', sm: '1.2rem', md: '1.5rem' },
              fontWeight: 700,
              color: '#001242',
              letterSpacing: 0.5,
              pb: { xs: 1, sm: 1.5, md: 2 },
            }}
          >
            AI Model Explanation
          </DialogTitle>
          <DialogContent
            sx={{
              p: { xs: 1.2, sm: 2, md: 3 },
              maxHeight: { xs: '60vh', sm: '65vh', md: '70vh' },
              overflowY: 'auto',
            }}
          >
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: { xs: 1.2, sm: 2, md: 3 } }}>
              {[
                {
                  name: "Gated Recurrent Unit (GRU)",
                  description: "A model that looks at someone's career step by step, in the order it happened. It helps recognize patterns over time and can predict when someone might be open to a new job based on their career history.",
                  useCase: "Ideal for: Understanding career progress and making time-based predictions"
                },
                {
                  name: "Extreme Gradient Boosting (XGBoost)",
                  description: "A model that combines many small decision trees to make strong predictions. It’s great at answering yes/no questions, like whether someone is likely to change jobs, and showing which factors matter most.",
                  useCase: "Ideal for: Predicting job changes and understanding key influencing factors"
                },
                {
                  name: "Temporal Fusion Transformer (TFT)",
                  description: "A very advanced model that can handle complex career data from different sources over time. It’s good at recognizing patterns even when there are many variables involved.",
                  useCase: "Ideal for: Analyzing complex career paths with multiple data points over time"
                }
              ].map((model, idx) => (
                <Box
                  key={idx}
                  sx={{
                    bgcolor: '#fff',
                    borderRadius: 2,
                    p: { xs: 1.2, sm: 2, md: 2.5 },
                    boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
                    minWidth: 0,
                    maxWidth: '100%',
                  }}
                >
                  <Typography sx={{ fontWeight: 700, color: '#001242', mb: 0.5, fontSize: { xs: '1rem', sm: '1.08rem', md: '1.15rem' } }}>
                    {model.name}
                  </Typography>
                  <Typography sx={{ color: '#666', mb: 0.5, fontSize: { xs: '0.88rem', sm: '0.95rem', md: '1rem' } }}>
                    {model.description}
                  </Typography>
                  <Typography sx={{ color: '#EB7836', fontSize: { xs: '0.85rem', sm: '0.92rem', md: '0.98rem' }, fontWeight: 600 }}>
                    {model.useCase}
                  </Typography>
                </Box>
              ))}
            </Box>
          </DialogContent>
          <DialogActions sx={{ p: { xs: 1, sm: 1.5, md: 2 } }}>
            <Button
              onClick={() => setShowModelInfo(false)}
              variant="contained"
              sx={{
                bgcolor: '#EB7836',
                color: '#fff',
                fontWeight: 700,
                fontSize: { xs: '0.92rem', sm: '1rem', md: '1.08rem' },
                letterSpacing: 0.5,
                textTransform: 'none',
                borderRadius: 2,
                boxShadow: '0 2px 8px #eb783664',
                px: { xs: 2, sm: 2.8, md: 3.4 },
                py: { xs: 1, sm: 1.2, md: 1.4 },
                '&:hover': { bgcolor: '#d97706' }
              }}
            >
              Continue
            </Button>
          </DialogActions>
        </Dialog>
        </Box>
      </Box>
      {/* Loading, Error, Prediction Results wie gehabt */}
      {loading && <LoadingSpinner sx={{ bgcolor: '#EB7836' }}/>}
      {error && (<Box sx={{ bgcolor: '#fff', borderRadius: '16px', p: '30px', boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)', mb: 4, color: '#FF2525', width: '100%' }}><Typography variant="h6" sx={{ mb: 1 }}>Error</Typography><Typography>{error}</Typography></Box>)}
      <div ref={predictionRef} />
      {prediction && (predictionModelType === 'tft' || predictionModelType === 'gru') && <PredictionResultTime prediction={prediction} />}
      {prediction && predictionModelType === 'xgboost' && <PredictionResultClassification prediction={prediction} />}
    </Box>
  );
};

export default ManualInput; 