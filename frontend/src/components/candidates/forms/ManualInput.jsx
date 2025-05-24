import React, { useState } from 'react';
import { Box, Typography, Button, TextField, Switch } from '@mui/material';
import PredictionResultTime from '../prediction/PredictionResultTime';
import PredictionResultClassification from '../prediction/PredictionResultClassification';
import DeleteOutlineIcon from '@mui/icons-material/DeleteOutline';

const modelOptions = [
  {
    value: 'gru',
    title: 'Gated Recurrent Unit (GRU)',
    description: 'Sequenzmodell für Zeitreihen und Karriereverläufe'
  },
  {
    value: 'xgboost',
    title: 'Extreme Gradient Boosting (XGBoost)',
    description: 'Leistungsstarkes Machine-Learning-Modell für strukturierte Daten'
  },
  {
    value: 'tft',
    title: 'Temporal Fusion Transformer (TFT)',
    description: 'Modernes Deep-Learning-Modell für komplexe Zeitreihen'
  }
];

const ManualInput = () => {
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
    setLoading(true);
    setError(null);
    setPrediction(null);

    try {
      const profile_data = {
        firstName: "Unbekannt",
        lastName: "Unbekannt",
        linkedinProfileInformation: JSON.stringify({
          firstName: "Unbekannt",
          lastName: "Unbekannt",
          workExperience: experiences.map(exp => ({
            company: exp.company || "",
            position: exp.position || "",
            startDate: exp.startDate ? formatDate(exp.startDate) : "",
            endDate: exp.endDate === 'Present' ? 'Present' : (exp.endDate ? formatDate(exp.endDate) : ""),
            type: "fullTime",
            location: "",
            description: ""
          })),
          education: education.map(edu => ({
            school: edu.school || "",
            degree: edu.degree || "",
            fieldOfStudy: edu.fieldOfStudy || "",
            startDate: edu.startDate ? formatDate(edu.startDate) : "",
            endDate: edu.endDate === 'Present' ? 'Present' : (edu.endDate ? formatDate(edu.endDate) : "")
          })),
          skills: [],
          location: "",
          headline: "",
          languageSkills: {}
        }),
        modelType: selectedModel.toLowerCase()
      };

      console.log("Sende Daten:", profile_data);

      const response = await fetch('http://localhost:5100/predict', {
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

  // Hilfsfunktion zum Formatieren der Daten
  const formatDate = (dateStr) => {
    if (!dateStr) return '';
    // Konvertiere YYYY-MM-DD zu MM/YYYY
    if (dateStr.includes('-')) {
      const [year, month, day]  = dateStr.split('-');
      return `${day}/${month}/${year}`;
    }
    return dateStr;
  };

  return (
    <Box sx={{ maxWidth: '1200px', ml: 0 }}>
      <Typography variant="h1" sx={{ fontSize: '2.5rem', fontWeight: 700, color: '#13213C', mb: 2 }}>Manuelle-Prognose</Typography>
      <Typography sx={{ color: '#666', mb: 4, fontSize: '1rem', maxWidth: '800px' }}>Analysieren Sie die Wechselwahrscheinlichkeit eines einzelnen Kandidaten basierend auf dessen Berufserfahrung.</Typography>
      <Box component="form" onSubmit={handleSubmit} sx={{ width: '100%' }}>
        {/* 
        <Box sx={{ bgcolor: '#fff', borderRadius: '18px', p: '0 0px 40px 0', boxShadow: '0 2px 8px rgba(0,0,0,0.04)', mb: 5 }}>
          <Box sx={{bgcolor: '#2C425C', borderTopLeftRadius: '18px',borderTopRightRadius: '18px',borderBottomLeftRadius: 0,borderBottomRightRadius: 0, p: '40px 0 40px 40px',boxShadow: '0 2px 8px rgba(0,0,0,0.04)',mb: 0 }}>
            <Typography variant="h2" sx={{ fontSize: '1.7rem', fontWeight: 700, color: '#FFF', mb: 1 }}>Ausbildung</Typography>
              <Typography sx={{ color: '#FFF', mb: 3, fontSize: '1.05rem' }}>
                Fügen Sie Informationen zur Ausbildung des Kandidaten hinzu.
              </Typography>
          </Box>
          <Box sx={{ bgcolor: '#fff', borderRadius: '12px', p: 3, mb: 2 }}>
            {education.map((edu, index) => (
              <Box
                key={index}
                sx={{
                  mb: 4,
                  bgcolor: '#fff',
                  borderRadius: '18px',
                  boxShadow: '0 2px 8px rgba(0,0,0,0.06)',
                  p: { xs: 2, sm: 4 },
                  border: '1px solid #f0f0f0',
                  position: 'relative'
                }}
              >
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                  <Typography sx={{ fontWeight: 700, color: '#2C425C', fontSize: '1.15rem' }}>
                    Ausbildung {index + 1}
                  </Typography>
                  {education.length > 1 && index > 0 && (
                    <Button
                      onClick={() => {
                        const newEducation = education.filter((_, i) => i !== index);
                        setEducation(newEducation);
                      }}
                      sx={{ color: '#FF2525',fontWeight: 600, fontSize: '1rem', textTransform: 'none', display: 'flex', alignItems: 'center', gap: 0.5, p: 0, minWidth: 0 }}startIcon={<DeleteOutlineIcon />} >
                      Entfernen
                    </Button>
                  )}
                </Box>
                <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr' }, gap: 3, mb: 2 }}>
                  <TextField
                    label="Schule/Hochschule"
                    value={edu.school}
                    onChange={(e) => handleEducationChange(index, 'school', e.target.value)}
                    sx={{ '& .MuiOutlinedInput-root': { fontSize: '1.1rem', minHeight: '58px' }, input: { fontSize: '1.1rem' } }}
                  />
                  <TextField
                    label="Abschluss"
                    value={edu.degree}
                    onChange={(e) => handleEducationChange(index, 'degree', e.target.value)}
                    sx={{ '& .MuiOutlinedInput-root': { fontSize: '1.1rem', minHeight: '58px' }, input: { fontSize: '1.1rem' } }}
                  />
                </Box>
                <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr' }, gap: 3, mb: 2 }}>
                  <TextField
                    label="Studienfach"
                    value={edu.fieldOfStudy}
                    onChange={(e) => handleEducationChange(index, 'fieldOfStudy', e.target.value)}
                    sx={{ gridColumn: '1 / -1', '& .MuiOutlinedInput-root': { fontSize: '1.1rem', minHeight: '58px' }, input: { fontSize: '1.1rem' } }}
                  />
                </Box>
                <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr' }, gap: 3, alignItems: 'center' }}>
                  <TextField
                    label="Startdatum"
                    type="date"
                    value={edu.startDate}
                    onChange={(e) => handleEducationChange(index, 'startDate', e.target.value)}
                    InputLabelProps={{ shrink: true }}
                    helperText={!edu.startDate ? "Bitte Startdatum wählen" : ""}
                    fullWidth
                    sx={{ '& .MuiOutlinedInput-root': { fontSize: '1.1rem', minHeight: '58px' }, input: { fontSize: '1.1rem' } }}
                  />
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                    <TextField
                      label="Enddatum"
                      type={edu.endDate === 'Present' ? 'text' : 'date'}
                      value={edu.endDate}
                      onChange={(e) => handleEducationChange(index, 'endDate', e.target.value)}
                      InputLabelProps={{ shrink: true }}
                      helperText={!edu.endDate ? "Bitte Enddatum wählen" : ""}
                      fullWidth
                      sx={{ '& .MuiOutlinedInput-root': { fontSize: '1.1rem', minHeight: '58px' }, input: { fontSize: '1.1rem' } }}
                    />
                    <Switch
                      checked={edu.endDate === 'Present'}
                      onChange={e => handleEducationChange(index, 'endDate', e.target.checked ? 'Present' : '')}
                      color="primary"
                    />
                    <Typography sx={{ fontSize: '1.1rem', fontWeight: 600, color: '#888', ml: 1 }}>
                      Present
                    </Typography>
                  </Box>
                </Box>
              </Box>
            ))}
            <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', mt: 6 }}>
              <Button onClick={handleAddEducation} fullWidth sx={{ bgcolor: '#fff',color: '#001B41',border: '2px dashed #001B41',borderRadius: '10px',fontWeight: 600,fontSize: '1rem',px: 4,py: 1.7,mt: 3,maxWidth: "100%", maxHeight: "50px",justifyContent: "center",alignItems: "center",display: "flex", margin: "0 auto", boxShadow: 'none', textTransform: 'none', transition: 'all 0.2s','&:hover': { bgcolor: '#fff', border: '2px solid #FF8000', color: '#FF8000'}}} >
                + WEITERE AUSBILDUNG HINZUFÜGEN
              </Button>
            </Box>
          </Box>
        </Box>
        */}
        <Box sx={{ bgcolor: '#fff', borderRadius: '14px', p: '0 0px 32px 0', boxShadow: '0 2px 8px rgba(0,0,0,0.04)', mb: 4 }}>
          <Box sx={{bgcolor: '#2C425C', borderTopLeftRadius: '14px',borderTopRightRadius: '14px',borderBottomLeftRadius: 0,borderBottomRightRadius: 0, p: '32px 0 32px 32px',boxShadow: '0 2px 8px rgba(0,0,0,0.04)',mb: 0 }}>
            <Typography variant="h2" sx={{ fontSize: '1.36rem', fontWeight: 700, color: '#fff', mb: 0.8 }}>Berufserfahrung</Typography>
            <Typography sx={{ color: '#fff', mb: 2.4, fontSize: '0.84rem' }}>
              Fügen Sie Informationen zur Berufserfahrung des Kandidaten hinzu.
            </Typography>
          </Box>
          <Box sx={{ bgcolor: '#fff', borderRadius: '9.6px', p: 2.4, mb: 1.6 }}>
            {experiences.map((exp, index) => (
              <Box
                key={index}
                sx={{ mb: 3.2,  bgcolor: '#fff', borderRadius: '14px', boxShadow: '0 2px 8px rgba(0,0,0,0.06)', p: { xs: 1.6, sm: 3.2 },border: '1px solid #f0f0f0',position: 'relative'}} >
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1.6 }}>
                  <Typography sx={{ fontWeight: 700, color: '#2C425C', fontSize: '0.92rem' }}>
                    Position {index + 1}
                  </Typography>
                  {experiences.length > 1 && index > 0 && (
                    <Button
                      onClick={() => handleRemoveExperience(index)}
                      sx={{ color: '#FF2525', fontWeight: 600, fontSize: '0.8rem', textTransform: 'none', display: 'flex',alignItems: 'center',  gap: 0.4,  p: 0, minWidth: 0 }} startIcon={<DeleteOutlineIcon />} >
                      Entfernen
                    </Button>
                  )}
                </Box>
                <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr' }, gap: 2.4, mb: 1.6 }}>
                  <TextField
                    label="Firma"
                    value={exp.company}
                    onChange={e => handleExperienceChange(index, 'company', e.target.value)}
                    sx={{ '& .MuiOutlinedInput-root': { fontSize: '0.88rem', minHeight: '46px' }, input: { fontSize: '0.88rem' } }}
                  />
                  <TextField
                    label="Position"
                    value={exp.position}
                    onChange={e => handleExperienceChange(index, 'position', e.target.value)}
                    sx={{ '& .MuiOutlinedInput-root': { fontSize: '0.88rem', minHeight: '46px' }, input: { fontSize: '0.88rem' } }}
                  />
                </Box>
                <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr' }, gap: 2.4, alignItems: 'center' }}>
                  <TextField
                    label="Startdatum"
                    type="date"
                    value={exp.startDate}
                    onChange={e => handleExperienceChange(index, 'startDate', e.target.value)}
                    InputLabelProps={{ shrink: true }}
                    helperText={!exp.startDate ? "Bitte Startdatum wählen" : ""}
                    fullWidth
                    sx={{ '& .MuiOutlinedInput-root': { fontSize: '0.88rem', minHeight: '46px' }, input: { fontSize: '0.88rem' } }}
                  />
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.6 }}>
                    <TextField
                      label="Enddatum"
                      type={exp.endDate === 'Present' ? 'text' : 'date'}
                      value={exp.endDate}
                      onChange={e => handleExperienceChange(index, 'endDate', e.target.value)}
                      InputLabelProps={{ shrink: true }}
                      helperText={!exp.endDate ? "Bitte Enddatum wählen" : ""}
                      fullWidth
                      sx={{ '& .MuiOutlinedInput-root': { fontSize: '0.88rem', minHeight: '46px' }, input: { fontSize: '0.88rem' } }}
                    />
                    <Switch
                      checked={exp.endDate === 'Present'}
                      onChange={e => handleExperienceChange(index, 'endDate', e.target.checked ? 'Present' : '')}
                      color="primary"
                    />
                    <Typography sx={{ fontSize: '0.88rem', fontWeight: 600, color: '#888', ml: 0.8 }}>
                      Present
                    </Typography>
                  </Box>
                </Box>
              </Box>
            ))}
            <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', mt: 4.8 }}>
              <Button onClick={handleAddExperience} fullWidth sx={{ bgcolor: '#fff',color: '#001B41',border: '2px dashed #001B41',borderRadius: '8px',fontWeight: 600,fontSize: '0.8rem',px: 3.2,py: 1.36,mt: 2.4,maxWidth: "100%", maxHeight: "40px",justifyContent: "center",alignItems: "center",display: "flex", margin: "0 auto", boxShadow: 'none', textTransform: 'none', transition: 'all 0.2s','&:hover': { bgcolor: '#fff', border: '2px solid #FF8000', color: '#FF8000'}}} >
                + WEITERE POSITION HINZUFÜGEN
              </Button>
            </Box>
          </Box>
        </Box>
        <Box sx={{ bgcolor: '#fff', borderRadius: '18px', p: '40px 30px', boxShadow: '0 2px 8px rgba(0,0,0,0.04)', mb: 5 }}>
          <Typography variant="h2" sx={{ fontSize: '1.7rem', fontWeight: 700, color: '#2C425C', mb: 1 }}>
            KI-Modell auswählen
          </Typography>
          <Typography sx={{ color: '#888', mb: 4, fontSize: '1.08rem' }}>
            Wählen Sie das passende Modell für eine präzise Prognose.
          </Typography>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, mb: 4 }}>
            {modelOptions.map(option => (
              <Box key={option.value}onClick={() => setSelectedModel(option.value)} sx={{cursor: 'pointer',bgcolor: '#fff', border: selectedModel === option.value ? '2px solid #FF8000' : '1.5px solid #e3e6f0', borderRadius: '16px', p: 3, boxShadow: selectedModel === option.value ? '0 2px 8px rgba(59,71,250,0.08)' : 'none', transition: 'all 0.2s', display: 'flex', flexDirection: 'column', outline: selectedModel === option.value ? '2px solid #FF8000' : 'none'}} >
                <Typography sx={{ fontWeight: 700, fontSize: '1.18rem', color: '#1a1a1a', mb: 0.5 }}>
                  {option.title}
                </Typography>
                <Typography sx={{ color: '#888', fontSize: '1.05rem' }}>
                  {option.description}
                </Typography>
              </Box>
            ))}
          </Box>
          <Box sx={{ display: 'flex', justifyContent: 'center', mt: 2 }}>
            <Button
              type="submit"
              disabled={loading || !selectedModel}
              sx={{ minWidth: 320, px: 4,  py: 1.8,  fontSize: '1.18rem', fontWeight: 700, borderRadius: '14px', color: '#fff', background: 'linear-gradient(90deg, #f4a65892 0%, #f4a65892 100%)', boxShadow: '0 4px 16px rgba(108,99,255,0.10)',textTransform: 'none', letterSpacing: 0.2, display: 'flex',  alignItems: 'center',justifyContent: 'center', gap: 1.5,'&:hover': {  background: 'linear-gradient(90deg, #FF8000 0%, #FF8000 100%)'}, '&.Mui-disabled': { background: '#e3e6f0',color: '#bdbdbd' } }}>
              Prognose starten
            </Button>
          </Box>
        </Box>   
      </Box>
      {loading && (<Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', my: 4 }}><Box sx={{ border: '3px solid #f3f3f3', borderTop: '3px solid #FF8000', borderRadius: '50%', width: '40px', height: '40px', animation: 'spin 1s linear infinite', '@keyframes spin': { '0%': { transform: 'rotate(0deg)' }, '100%': { transform: 'rotate(360deg)' } } }} /></Box>)}
      {error && (<Box sx={{ bgcolor: '#fff', borderRadius: '16px', p: '30px', boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)', mb: 4, color: '#FF2525', width: '100%' }}><Typography variant="h6" sx={{ mb: 1 }}>Fehler</Typography><Typography>{error}</Typography></Box>)}
      {prediction && selectedModel === 'tft' && (<><PredictionResultTime prediction={prediction} /></>)}
      {prediction && selectedModel === 'gru' && <PredictionResultTime prediction={prediction} />}
      {prediction && selectedModel === 'xgboost' && <PredictionResultClassification prediction={prediction} />}
    </Box>
  );
};

export default ManualInput; 