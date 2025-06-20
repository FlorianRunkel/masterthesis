import React, { useState, useRef, useEffect } from 'react';
import { Box, Typography, Button, Alert } from '@mui/material';
import ResultsTableClassification from '../components/display/table_classification';
import LoadingSpinner from '../components/shared/loading_spinner';
import ResultsTableTimeSeries from '../components/display/table_timeseries';
import { useTheme } from '@mui/material/styles';
import { useMediaQuery } from '@mui/material';
import InsertDriveFileOutlinedIcon from '@mui/icons-material/InsertDriveFileOutlined';
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';
import WarningAmberOutlinedIcon from '@mui/icons-material/WarningAmberOutlined';
import CloudUploadOutlinedIcon from '@mui/icons-material/CloudUploadOutlined';

const modelOptions = [
  {
    value: 'gru',
    title: 'Gated Recurrent Unit (GRU)',
    description: 'Sequence model for time series and career trajectories'
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

const BatchUpload = () => {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [results, setResults] = useState(null);
  const [isSaving, setIsSaving] = useState(false);
  const [saveSuccess, setSaveSuccess] = useState(false);
  const [saveError, setSaveError] = useState(null);
  const [modelType, setModelType] = useState('');
  const [originalProfiles, setOriginalProfiles] = useState([]);
  const [showModelChangeHint, setShowModelChangeHint] = useState(false);
  const [resultsModelType, setResultsModelType] = useState('');
  const [dragActive, setDragActive] = useState(false);
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const resultsRef = useRef(null);

  useEffect(() => {
    if (results && resultsRef.current) {
      resultsRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [results]);

  useEffect(() => {
    if (results) {
      localStorage.setItem('batchResults', JSON.stringify(results));
      localStorage.setItem('batchResultsModelType', resultsModelType);
    }
  }, [results, resultsModelType]);

  useEffect(() => {
    const saved = localStorage.getItem('batchResults');
    const savedType = localStorage.getItem('batchResultsModelType');
    if (saved) {
      setResults(JSON.parse(saved));
      setResultsModelType(savedType || '');
    }
  }, []);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (!selectedFile) return;
    if (!selectedFile.name.endsWith('.csv')) {
      alert("Please select a valid CSV file.");
      return;
    }
    setFile(selectedFile);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileChange({ target: { files: e.dataTransfer.files } });
    }
  };

  const handleUpload = async () => {
    setShowModelChangeHint(false);
    if (!file) {
      alert("Please select a CSV file.");
      return;
    }
    setResults(null);
    setOriginalProfiles([]);
    setResultsModelType(modelType);
    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);
    formData.append('modelType', modelType);
    try {
      const response = await fetch('/predict-batch', {
        method: 'POST',
        body: formData
      });
      if (!response.ok) {
        throw new Error('Upload failed');
      }
      const data = await response.json();
      if (data.error) {
        setResults({
          error: data.error,
          message: "Please check the format of your CSV file."
        });
        return;
      }
      setResults(data.results);
      if (data.originalProfiles) setOriginalProfiles(data.originalProfiles);
    } catch (error) {
      setError(error.message);
      setResults({
        error: error.message,
        message: "Please make sure your CSV file contains the following columns:",
        requirements: [
          "firstName (first name)",
          "lastName (last name)",
          "linkedinProfile (LinkedIn-URL)",
          "positions (experience in JSON format)"
        ]
      });
    } finally {
      setLoading(false);
    }
  };

  const handleSaveCandidates = async (candidates) => {
    setIsSaving(true);
    setSaveError(null);
    setSaveSuccess(false);

    try {
      const user = JSON.parse(localStorage.getItem('user'));
      const uid = user?.uid;

      // Hole für alle Kandidaten mit LinkedIn-Link die Profildaten
      const candidatesWithProfile = await Promise.all(
        candidates.map(async (candidate) => {
          if (candidate.linkedinProfile) {
            try {
              const response = await fetch('/scrape-linkedin', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ url: candidate.linkedinProfile }),
              });
              const data = await response.json();
              if (data && !data.error) {
                // Profildaten extrahieren und auf oberster Ebene speichern
                const [firstName, ...rest] = (data.name || '').split(' ');
                const lastName = rest.join(' ');
                return {
                  ...candidate,
                  firstName: firstName || '',
                  lastName: lastName || '',
                  currentPosition: data.currentTitle || '',
                  imageUrl: data.imageUrl || '',
                  experience: data.experience || [],
                  location: data.location || '',
                  industry: data.industry || '',
                  linkedinProfileInformation: JSON.stringify(data),
                };
              }
            } catch (err) {
              // Fehler beim Scrapen: Kandidat trotzdem speichern, aber mit Fehlerhinweis
              return {
                ...candidate,
                scrapeError: 'LinkedIn scraping failed',
              };
            }
          }
          // Falls kein LinkedIn-Link, Kandidat unverändert zurückgeben
          return candidate;
        })
      );

      // Jetzt alle Kandidaten speichern
      const candidatesWithModel = candidatesWithProfile.map(candidate => ({
        ...candidate,
        modelType: modelType
      }));

      const response = await fetch('/api/candidates', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-User-Uid': uid,
        },
        body: JSON.stringify(candidatesWithModel),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || 'Error saving candidates');
      }
      setSaveSuccess(true);
      setResults(null); // Reset results after successful save
      setFile(null);
    } catch (error) {
      setSaveError(error.message);
    } finally {
      setIsSaving(false);
    }
  };

  const handleModelChange = (value) => {
    setModelType(value);
    setShowModelChangeHint(true);
  };

  return (
    <Box sx={{ maxWidth: '1200px',  marginLeft: isMobile ? 0 : '240px' }}>

      <Typography variant="h1" sx={{ 
        fontSize: isMobile ? '1.8rem' : '2.5rem', 
        fontWeight: 700, 
        color: '#001242', 
        mb: 2 
      }}>
        Batch Upload
      </Typography>
      <Typography sx={{ 
        color: '#666', 
        mb: 4, 
        fontSize: isMobile ? '0.9rem' : '1rem', 
        maxWidth: '800px' 
      }}>
        Upload a CSV file to analyze the job change probability of multiple candidates at once.
      </Typography>
      
      {/* Upload-Box */}
      <Box sx={{ bgcolor: '#fff', borderRadius: '12.8px', p: '24px', boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)', mb: 3.2 }}>
        <Typography sx={{ fontWeight: 700, color: '#001242', fontSize: '1.1rem', mb: 1.2, display: 'flex', alignItems: 'center', gap: 1 }}>
          <CloudUploadOutlinedIcon sx={{ color: '#001242', fontSize: 24, mr: 1 }} />
          Candidates CSV-File Upload
        </Typography>
        <Typography sx={{ color: '#888', fontSize: '0.95rem', mb: 2 }}>
          Upload a CSV file to analyze the job change probability of multiple candidates at once.
        </Typography>
        <Box
          sx={{
            border: dragActive ? '2px solid #EB7836' : '2px dashed #bdbdbd',
            borderRadius: '12px',
            bgcolor: '#F8F9FB',
            p: 4,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            mb: 2
          }}
          onDragOver={handleDragOver}
          onDragEnter={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <InsertDriveFileOutlinedIcon sx={{ color: '#bdbdbd', fontSize: 48, mb: 1 }} />
          <Typography sx={{ color: '#888', fontSize: '0.95rem'}}>Drop your CSV file here or click to upload</Typography>
          <Typography sx={{ fontSize: '0.8rem', color: '#888', textAlign: 'center', mb: '10px' }}>{file ? file.name : 'Supported formats: .csv'}</Typography>
          <Button
            component="label"
            htmlFor="csvFile"
            sx={{
              bgcolor: '#fff',
              color: '#001B41',
              border: '1.6px solid #bdbdbd',
              borderRadius: '9.6px',
              fontSize: '0.88rem',
              fontWeight: 700,
              cursor: 'pointer',
              textTransform: 'none',
              px: 2,
              py: 1,
              boxShadow: 'none',
              '&:hover': {
                bgcolor: '#fff',
                border: '1.6px solid #EB7836',
                color: '#EB7836',
              },
            }}
          >
            Select File
            <input type="file" id="csvFile" accept=".csv" onChange={handleFileChange} style={{ display: 'none' }} />
          </Button>
         </Box>
      </Box>

      <Box sx={{ bgcolor: '#fff', borderRadius: '12.8px', p: '24px', boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)', mb: 3.2 }}>
        <Typography sx={{ fontWeight: 700, color: '#001242', fontSize: '1.1rem', mb: 1.2, display: 'flex', alignItems: 'center', gap: 1 }}>
          <WarningAmberOutlinedIcon sx={{ color: '#FFB300', fontSize: 22, mr: 1 }} />
          CSV Format & Requirements
        </Typography>
        <Typography sx={{ color: '#888', fontSize: '0.95rem', mb: 2 }}>
          The CSV file must contain the following columns in the first row:
        </Typography>
        <Box sx={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 1, border: '1px solid #bdbdbd', borderRadius: '10px', px: 1, py: 0.8, fontSize: { xs: '0.78rem', sm: '0.95rem' }, fontWeight: 600, background: '#eee', maxWidth: '100%', mb: 1 }}>
          <Box sx={{
            bgcolor: '#000',
            color: '#fff',
            borderRadius: '16px',
            px: 1.2,
            py: 0.1,
            fontSize: { xs: '0.78rem', sm: '0.92rem' },
            fontWeight: 600,
            height: 24,
            display: 'flex',
            alignItems: 'center',
            mb: { xs: 0.5, sm: 0 },
          }}>
            Required
          </Box>
          <Box sx={{
            bgcolor: '#F8F9FB',
            border: '1px solid #bdbdbd',
            borderRadius: '6px',
            px: 1.2,
            py: 0.1,
            fontSize: { xs: '0.78rem', sm: '0.95rem' },
            fontWeight: 600,
            height: 24,
            display: 'flex',
            alignItems: 'center',
            mb: { xs: 0.5, sm: 0 },
          }}>
            firstName
          </Box>
          <Typography sx={{
            fontSize: { xs: '0.78rem', sm: '0.95rem' },
            color: '#000',
            alignSelf: 'center',
            mx: 1,
            flex: 1,
            minWidth: 120,
            mb: { xs: 0.5, sm: 0 },
            wordBreak: 'break-word',
          }}>
            First Name of Candidate
          </Typography>
        </Box>
        <Box sx={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 1, border: '1px solid #bdbdbd', borderRadius: '10px', px: 1, py: 0.8, fontSize: { xs: '0.78rem', sm: '0.95rem' }, fontWeight: 600, background: '#eee', maxWidth: '100%', mb: 1 }}>
          <Box sx={{
            bgcolor: '#000',
            color: '#fff',
            borderRadius: '16px',
            px: 1.2,
            py: 0.1,
            fontSize: { xs: '0.78rem', sm: '0.92rem' },
            fontWeight: 600,
            height: 24,
            display: 'flex',
            alignItems: 'center',
            mb: { xs: 0.5, sm: 0 },
          }}>
            Required
          </Box>
          <Box sx={{
            bgcolor: '#F8F9FB',
            border: '1px solid #bdbdbd',
            borderRadius: '6px',
            px: 1.2,
            py: 0.1,
            fontSize: { xs: '0.78rem', sm: '0.95rem' },
            fontWeight: 600,
            height: 24,
            display: 'flex',
            alignItems: 'center',
            mb: { xs: 0.5, sm: 0 },
          }}>
            lastName
          </Box>
          <Typography sx={{
            fontSize: { xs: '0.78rem', sm: '0.95rem' },
            color: '#000',
            alignSelf: 'center',
            mx: 1,
            flex: 1,
            minWidth: 120,
            mb: { xs: 0.5, sm: 0 },
            wordBreak: 'break-word',
          }}>
            Last Name of Candidate
          </Typography>
        </Box>
        <Box sx={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 1, border: '1px solid #bdbdbd', borderRadius: '10px', px: 1, py: 0.8, fontSize: { xs: '0.78rem', sm: '0.95rem' }, fontWeight: 600, background: '#eee', maxWidth: '100%', mb: 1 }}>
          <Box sx={{
            bgcolor: '#000',
            color: '#fff',
            borderRadius: '16px',
            px: 1.2,
            py: 0.1,
            fontSize: { xs: '0.78rem', sm: '0.92rem' },
            fontWeight: 600,
            height: 24,
            display: 'flex',
            alignItems: 'center',
            mb: { xs: 0.5, sm: 0 },
          }}>
            Required
          </Box>
          <Box sx={{
            bgcolor: '#F8F9FB',
            border: '1px solid #bdbdbd',
            borderRadius: '6px',
            px: 1.2,
            py: 0.1,
            fontSize: { xs: '0.78rem', sm: '0.95rem' },
            fontWeight: 600,
            height: 24,
            display: 'flex',
            alignItems: 'center',
            mb: { xs: 0.5, sm: 0 },
          }}>
            profileLink
          </Box>
          <Typography sx={{
            fontSize: { xs: '0.78rem', sm: '0.95rem' },
            color: '#000',
            alignSelf: 'center',
            mx: 1,
            flex: 1,
            minWidth: 120,
            mb: { xs: 0.5, sm: 0 },
            wordBreak: 'break-word',
          }}>
            LinkedIn Profile URL
          </Typography>
        </Box>
        <Box sx={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 1, border: '1px solid #bdbdbd', borderRadius: '10px', px: 1, py: 0.8, fontSize: { xs: '0.78rem', sm: '0.95rem' }, fontWeight: 600, background: '#eee', maxWidth: '100%', mb: 1 }}>
          <Box sx={{
            bgcolor: '#000',
            color: '#fff',
            borderRadius: '16px',
            px: 1.2,
            py: 0.1,
            fontSize: { xs: '0.78rem', sm: '0.92rem' },
            fontWeight: 600,
            height: 24,
            display: 'flex',
            alignItems: 'center',
            mb: { xs: 0.5, sm: 0 },
          }}>
            Required
          </Box>
          <Box sx={{
            bgcolor: '#F8F9FB',
            border: '1px solid #bdbdbd',
            borderRadius: '6px',
            px: 1.2,
            py: 0.1,
            fontSize: { xs: '0.78rem', sm: '0.95rem' },
            fontWeight: 600,
            height: 24,
            display: 'flex',
            alignItems: 'center',
            mb: { xs: 0.5, sm: 0 },
          }}>
            linkedinProfileInformation
          </Box>
          <Typography sx={{
            fontSize: { xs: '0.78rem', sm: '0.95rem' },
            color: '#000',
            alignSelf: 'center',
            mx: 1,
            flex: 1,
            minWidth: 120,
            mb: { xs: 0.5, sm: 0 },
            wordBreak: 'break-word',
          }}>
            JSON-String with LinkedIn Information
          </Typography>
          <Box sx={{ bgcolor: '#FF2525', color: '#fff', borderRadius: '16px', fontSize: '0.7rem', fontWeight: 700, px: 1.2, py: 0.1, height: 24, display: 'flex', alignItems: 'center' }}>IMPORTANT</Box>
        </Box>
        <Typography sx={{ color: '#888', fontSize: '0.95rem', mb: 1 , mt: 3 }}>
          <InfoOutlinedIcon sx={{ color: '#888', fontSize: 18, mr: 0.5, mb: '2px' }} />
          The structure of <b style={{ color: '#000000' }}>linkedinProfileInformation</b> must exactly match the following example:
        </Typography>
        <Box sx={{ bgcolor: '#000000', color: '#fff', borderRadius: '8px', p: 2, fontSize: '0.7rem', fontFamily: 'monospace', overflowX: 'auto', mb: 2, whiteSpace: 'pre', height: '350px'}}>
          {`{
  "skills": ["Python", "SQL", "Data Analysis"],
  "firstName": "Florian",
  "lastName": "Runkel",
  "profilePicture": "https://media.licdn.com/dms/image/v2/D4D03AQFzqblQQVoUsA/profile-displayphoto-shrink_400_400/profile-displayphoto-shrink_400_400/0/1723124243698?e=1755734400&v=beta&t=PfZPoNjZkF-e0Bhy-llpyagYBTsERaauYw5-cDpjk3I",
  "linkedinProfile": "https://www.linkedin.com/in/florian-runkel-82b521228/",
  "education": [
    {
      "duration": "01/10/2023 - 01/10/2025",
      "institution": "University Regensburg",
      "endDate": "01/10/2025",
      "degree": "Master of Science - MS, Information Systems",
      "location": "Regensburg, Bavaria, Germany",
      "subjectStudy": "Information Systems",
      "startDate": "01/10/2023"
    }
  ],
  "workExperience": [
    {
      "duration": "17/06/2023 - Present",
      "endDate": "Present",
      "company": "aurio Technology GmbH",
      "location": "Munich, Bayern, Deutschland",
      "position": "Working Student",
      "type": "fullTime",
      "startDate": "17/06/2023"
    }
  ],
  "location": "Munich, Bavaria, Germany",
  "headline": "M.Sc. Information Systems @UR | Founders Associate Tech & AI @aurio",
  "languageSkills": {
    "Deutsch": "Native or bilingual proficiency",
    "English": "Fluent"
  }
}`}
        </Box>
        <Button
          variant="outlined"
          sx={{ border: '1.6px solid #001B41', color: '#001B41', fontWeight: 600, fontSize: '0.8rem', borderRadius: '9.6px', textTransform: 'none', '&:hover': { border: '1.6px solid #EB7836', color: '#EB7836', background: '#fff' } }}
          href="/testfile.csv"
          download
        >
          Download Example CSV
        </Button>
      </Box>

      {/* Modellauswahl-Box */}
      <Box sx={{ bgcolor: '#fff', borderRadius: '14px', p: '32px 24px', boxShadow: '0 2px 8px rgba(0,0,0,0.08)', mb: 1.6 }}>
        <Typography variant="h2" sx={{ fontSize: '1.36rem', fontWeight: 700, color: '#001242', mb: 0.8 }}>
          Select AI model
        </Typography>
        <Typography sx={{ color: '#888', mb: 3.2, fontSize: '0.86rem' }}>
          Select the appropriate model for a precise prediction.
        </Typography>
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.6, mb: 1.6 }}>
          {modelOptions.map(option => (
            <Box key={option.value} onClick={() => handleModelChange(option.value)} sx={{ cursor: 'pointer', bgcolor: '#fff', border: modelType === option.value ? '1.6px solid #EB7836' : '1.2px solid #e3e6f0', borderRadius: '12.8px', p: 2.4, boxShadow: modelType === option.value ? '0 2px 8px rgba(59,71,250,0.08)' : 'none', transition: 'all 0.2s', display: 'flex', flexDirection: 'column', outline: modelType === option.value ? '1.6px solid #EB7836' : 'none', mb: 0.8 }}>
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
          onClick={handleUpload}
          disabled={!file || loading || !modelType}
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
            mb: 6,
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

      {error && (
        <Alert severity="error" sx={{ mt: 2 }}>
          {error}
        </Alert>
      )}
      {saveError && (
        <Alert severity="error" sx={{ mt: 2 }}>
          {saveError}
        </Alert>
      )}
      {saveSuccess && (
        <Alert severity="success" sx={{ mt: 2 }}>
          Candidates were successfully saved!
        </Alert>
      )}
      {loading && <LoadingSpinner />}
      {results && !loading && (
        <Box ref={resultsRef} sx={{ bgcolor: '#fff', borderRadius: '14px', boxShadow: '0 2px 8px rgba(0,0,0,0.08)', mb: 1.6 }}>
          {resultsModelType === 'tft' || resultsModelType === 'gru' ? (
            <ResultsTableTimeSeries
              results={results}
              onSave={handleSaveCandidates}
              isSaving={isSaving}
              originalProfiles={originalProfiles}
            />
          ) : (
            <ResultsTableClassification
              results={results}
              onSave={handleSaveCandidates}
              isSaving={isSaving}
              originalProfiles={originalProfiles}
            />
          )}
        </Box>
      )}
    </Box>
  );
};

export default BatchUpload; 