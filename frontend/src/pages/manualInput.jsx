import React, { useState, useRef, useEffect } from 'react';
import { Box, Typography, Button, TextField, Select, MenuItem, Checkbox, FormControlLabel } from '@mui/material';
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
  const [experiences, setExperiences] = useState([{
    company: '',
    position: '',
    startDate: '',
    endDate: ''
  }]);
  const [selectedModel, setSelectedModel] = useState('');
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [education, setEducation] = useState([{
    school: '',
    degree: '',
    fieldOfStudy: '',
    startDate: '',
    endDate: ''
  }]);
  const [showModelChangeHint, setShowModelChangeHint] = useState(false);
  const [predictionModelType, setPredictionModelType] = useState('');
  const predictionRef = useRef(null);

  useEffect(() => {
    if (prediction && predictionRef.current) {
      predictionRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [prediction]);

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
    
        // dateStr is in YYYY-MM-DD format from the date picker
        if (dateStr.includes('-')) {
            const [year, month, day] = dateStr.split('-');
            
            if (model === 'tft') {
                return `${day}/${month}/${year}`; // Format for TFT: DD/MM/YYYY
            } else {
                return `${month}/${year}`; // Format for others: MM/YYYY
            }
        }
        
        return dateStr; // Fallback
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
  };


  return (
    <Box sx={{ p: 0, m: 0 }}>
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
        Analyze the job change probability of a single candidate based on their work experience.
      </Typography>
      <Box component="form" onSubmit={handleSubmit} sx={{ width: '100%' }}>

        {/* Education Section */}
        <Paper sx={{ p: { xs: 2, sm: 3 }, mb: 4, borderRadius: '16px', boxShadow: '0 8px 16px rgba(0,0,0,0.05)' }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: education.length > 0 ? 3 : 0 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
                    <SchoolIcon sx={{ color: '#001242' }} />
                    <Typography variant="h6" sx={{ fontWeight: 600, color: '#001242' }}>Education</Typography>
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
                    Education {index + 1}
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

        {/* Work Experience Section */}
        <Paper sx={{ p: { xs: 2, sm: 3 }, mb: 4, borderRadius: '16px', boxShadow: '0 8px 16px rgba(0,0,0,0.05)' }}>
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
                    Position {index + 1}
                  </Typography>
                  {index > 0 && (
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
        
        <Box sx={{ bgcolor: '#fff', borderRadius: '14px', p: '32px 24px', boxShadow: '0 2px 8px rgba(0,0,0,0.08)', mb: 1.6 }}>
          <Typography variant="h2" sx={{ fontSize: '1.36rem', fontWeight: 700, color: '#001242', mb: 0.8 }}>
            Select AI model
          </Typography>
          <Typography sx={{ color: '#888', mb: 3.2, fontSize: '0.86rem' }}>
            Select the appropriate model for a precise prediction.
          </Typography>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.6, mb: 1.6 }}>
            {modelOptions.map(option => (
              <Box key={option.value} onClick={() => handleModelChange(option.value)} sx={{cursor: 'pointer',bgcolor: '#fff', border: selectedModel === option.value ? '2px solid #EB7836' : '1.5px solid #e3e6f0', borderRadius: '16px', p: 3, boxShadow: selectedModel === option.value ? '0 2px 8px rgba(59,71,250,0.08)' : 'none', transition: 'all 0.2s', display: 'flex', flexDirection: 'column', outline: selectedModel === option.value ? '2px solid #EB7836' : 'none'}} >
                <Typography sx={{ fontWeight: 700, fontSize: '0.94rem', color: '#1a1a1a', mb: 0.4 }}>
                  {option.title}
                </Typography>
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
        </Box>
        <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 2 }}>
            <Button
              type="submit"
              disabled={loading || !selectedModel}
              sx={{
                minWidth: 256,
                px: 3.2,
                py: 1.44,
                fontSize: '0.94rem',
                fontWeight: 700,
                borderRadius: '11.2px',
                color: '#fff',
                background: 'linear-gradient(90deg, #EB7836 0%, #EB7836 100%)',
                boxShadow: '0 4px 16px rgba(108,99,255,0.10)',
                textTransform: 'none',
                letterSpacing: 0.16,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: 1.2,
                mt: 1.6,
                mb: 5,
                '&:hover': {
                  background: 'linear-gradient(90deg, #EB7836 0%, #EB7836 100%)',
                },
                '&.Mui-disabled': {
                  background: '#e3e6f0',
                  color: '#bdbdbd',
                },
              }}
            >
              Start prediction
            </Button>
          </Box>
      </Box>
      {loading && (<Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', my: 4 }}><Box sx={{ border: '3px solid #f3f3f3', borderTop: '3px solid #EB7836', borderRadius: '50%', width: '40px', height: '40px', animation: 'spin 1s linear infinite', '@keyframes spin': { '0%': { transform: 'rotate(0deg)' }, '100%': { transform: 'rotate(360deg)' } } }} /></Box>)}
      {error && (<Box sx={{ bgcolor: '#fff', borderRadius: '16px', p: '30px', boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)', mb: 4, color: '#FF2525', width: '100%' }}><Typography variant="h6" sx={{ mb: 1 }}>Error</Typography><Typography>{error}</Typography></Box>)}
      <div ref={predictionRef} />
      {prediction && predictionModelType === 'tft' && (<><PredictionResultTime prediction={prediction} /></>)}
      {prediction && predictionModelType === 'gru' && <PredictionResultTime prediction={prediction} />}
      {prediction && predictionModelType === 'xgboost' && <PredictionResultClassification prediction={prediction} />}
    </Box>
  );
};

export default ManualInput; 