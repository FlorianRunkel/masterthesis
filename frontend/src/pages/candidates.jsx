import React, { useState, useEffect } from 'react';
import { Box, Typography, TextField, IconButton, Button } from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import InputAdornment from '@mui/material/InputAdornment';
import CandidateCard from '../components/display/candidate_card';
import LoadingSpinner from '../components/shared/loading_spinner';
import { useTheme } from '@mui/material/styles';
import { useMediaQuery } from '@mui/material';
import { API_BASE_URL } from '../api';
import axios from 'axios';

const CandidatesPage = () => {
  const [candidates, setCandidates] = useState([]);
  const [filteredCandidates, setFilteredCandidates] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedModel, setSelectedModel] = useState('all');
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));

  /*
  Fetch candidates.
  */
  useEffect(() => {
    const fetchCandidates = async () => {
      try {
        setLoading(true);
        const response = await axios.get(`${API_BASE_URL}/api/candidates`);
        const data = response.data;

        // UID des eingeloggten Users holen
        const user = JSON.parse(localStorage.getItem('user'));
        const uid = user?.uid;

        // Only show candidates with matching UID
        const myCandidates = data.filter(candidate => candidate.uid === uid);

        setCandidates(myCandidates);
        setFilteredCandidates(myCandidates);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchCandidates();
  }, []);

  /*
  Filter candidates.
  */
  useEffect(() => {
    const filtered = candidates.filter(candidate => {
      const name = `${candidate.firstName || ''} ${candidate.lastName || ''}`.toLowerCase();
      const matchesSearch = name.includes(searchTerm.toLowerCase()) || 
                           candidate.linkedinProfile?.toLowerCase().includes(searchTerm.toLowerCase());
      const matchesModel = selectedModel === 'all' || candidate.modelType === selectedModel;

      return matchesSearch && matchesModel;
    });
    setFilteredCandidates(filtered);
  }, [searchTerm, selectedModel, candidates]);

  /*
  Handle delete candidate.
  */
  const handleDeleteCandidate = async (candidateId) => {
    try {
      const user = JSON.parse(localStorage.getItem('user'));
      const uid = user?.uid;

      if (!uid) {
        throw new Error("User not authenticated");
      }

      const response = await axios.delete(`${API_BASE_URL}/api/candidates/${candidateId}`, {
        headers: {
          'X-User-Uid': uid,
        },
      });
      setCandidates(prevCandidates => prevCandidates.filter(c => c._id !== candidateId));
    } catch (err) {
      setError(err.message);
    }
  };

  if (loading) {
    return <LoadingSpinner />;
  }

  if (error) {
    return (
      <Box sx={{margin: '0 auto', p: 4 }}>
        <Typography variant="h2" sx={{ mb: 3, color: '#1a1a1a', fontSize: '1.5rem', fontWeight: 600 }}>
          Error
        </Typography>
        <Typography sx={{ color: '#dc3545' }}>{error}</Typography>
      </Box>
    );
  }

  return (
    <Box>
      <Typography variant="h1" sx={{ 
        fontSize: isMobile ? '1.8rem' : '2.5rem', 
        fontWeight: 700, 
        color: '#001242', 
        mb: 2 
      }}>
        Candidates
      </Typography>
      <Typography sx={{ 
        color: '#666', 
        mb: 4, 
        fontSize: isMobile ? '0.9rem' : '1rem', 
        maxWidth: '800px' 
      }}>
        Here you can see all saved candidates.
      </Typography>
      <Box sx={{
        bgcolor: '#fff',
        borderRadius: '13px',
        p: '24px',
        boxShadow: '0 2px 8px rgba(0,0,0,0.05)',
        mb: 3.2,
        width: '100%'
      }}>
        <Typography variant="h2" sx={{
          fontSize: '1.2rem',
          fontWeight: 600,
          color: '#1a1a1a',
          mb: 2.4
        }}>
          Search candidates
        </Typography>
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.6 }}>
          <TextField 
            placeholder="Search by name or LinkedIn profile..."
            variant="outlined"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            sx={{ 
              width: '100%',
              '& .MuiOutlinedInput-root': {
                bgcolor: '#fff',
                borderRadius: '9.6px',
                '& fieldset': {
                  borderColor: '#001B41',
                  borderWidth: 1.6,
                  transition: 'all 0.3s ease'
                },
                '&:hover fieldset': {
                  borderColor: '#001B41',
                  borderWidth: 1.6
                },
                '&.Mui-focused fieldset': {
                  borderColor: '#001B41',
                  borderWidth: 1.6
                }
              }
            }}
            InputProps={{
              endAdornment: (
                <InputAdornment position="end">
                  <IconButton sx={{ color: '#001B41', '&:hover': { color: '#EB7836' } }}>
                    <SearchIcon />
                  </IconButton>
                </InputAdornment>
              )
            }}
          />
          <Box sx={{ 
            display: 'flex', 
            gap: 1.6, 
            flexWrap: 'wrap',
          }}>
            <Button
              variant={selectedModel === 'all' ? 'contained' : 'outlined'}
              onClick={() => setSelectedModel('all')}
              sx={{
                bgcolor: selectedModel === 'all' ? '#001B41' : 'transparent',
                color: selectedModel === 'all' ? '#fff' : '#001B41',
                height: '36px',
                minWidth: '96px',
                borderRadius: '6.4px',
                border: '1.6px solid #001B41',
                fontSize: '0.8rem',
                transition: 'all 0.3s ease',
                '&:hover': {
                  bgcolor: selectedModel === 'all' ? '#EB7836' : '#fff',
                  borderColor: '#EB7836',
                  color: selectedModel === 'all' ? '#fff' : '#EB7836',
                  transform: 'translateY(-1.6px)',
                  boxShadow: '0 2px 8px rgba(0,0,0,0.05)'
                }
              }}
            >
              All Models
            </Button>
            <Button
              variant={selectedModel === 'gru' ? 'contained' : 'outlined'}
              onClick={() => setSelectedModel('gru')}
              sx={{
                bgcolor: selectedModel === 'gru' ? '#001B41' : 'transparent',
                color: selectedModel === 'gru' ? '#fff' : '#001B41',
                height: '36px',
                minWidth: '96px',
                borderRadius: '6.4px',
                border: '1.6px solid #001B41',
                fontSize: '0.8rem',
                transition: 'all 0.3s ease',
                '&:hover': {
                  bgcolor: selectedModel === 'gru' ? '#EB7836' : '#fff',
                  borderColor: '#EB7836',
                  color: selectedModel === 'gru' ? '#fff' : '#EB7836',
                  transform: 'translateY(-1.6px)',
                  boxShadow: '0 2px 8px rgba(0,0,0,0.05)'
                }
              }}
            >
              GRU
            </Button>
            <Button
              variant={selectedModel === 'tft' ? 'contained' : 'outlined'}
              onClick={() => setSelectedModel('tft')}
              sx={{
                bgcolor: selectedModel === 'tft' ? '#001B41' : 'transparent',
                color: selectedModel === 'tft' ? '#fff' : '#001B41',
                height: '36px',
                minWidth: '96px',
                borderRadius: '6.4px',
                border: '1.6px solid #001B41',
                fontSize: '0.8rem',
                transition: 'all 0.3s ease',
                '&:hover': {
                  bgcolor: selectedModel === 'tft' ? '#EB7836' : '#fff',
                  borderColor: '#EB7836',
                  color: selectedModel === 'tft' ? '#fff' : '#EB7836',
                  transform: 'translateY(-1.6px)',
                  boxShadow: '0 2px 8px rgba(0,0,0,0.05)'
                }
              }}
            >
              TFT
            </Button>
            <Button
              variant={selectedModel === 'xgboost' ? 'contained' : 'outlined'}
              onClick={() => setSelectedModel('xgboost')}
              sx={{
                bgcolor: selectedModel === 'xgboost' ? '#001B41' : 'transparent',
                color: selectedModel === 'xgboost' ? '#fff' : '#001B41',
                height: '36px',
                minWidth: '96px',
                borderRadius: '6.4px',
                border: '1.6px solid #001B41',
                fontSize: '0.8rem',
                transition: 'all 0.3s ease',
                '&:hover': {
                  bgcolor: selectedModel === 'xgboost' ? '#EB7836' : '#fff',
                  borderColor: '#EB7836',
                  color: selectedModel === 'xgboost' ? '#fff' : '#EB7836',
                  transform: 'translateY(-1.6px)',
                  boxShadow: '0 2px 8px rgba(0,0,0,0.05)'
                }
              }}
            >
              XGBoost
            </Button>
          </Box>
        </Box>
      </Box>
      {filteredCandidates.length === 0 ? (
        <Box sx={{
          bgcolor: '#fff',
          borderRadius: '16px',
          p: '24px',
          boxShadow: '0 2px 8px rgba(0,0,0,0.05)',
          textAlign: 'center',
          width: '100%'
        }}>
          <Typography sx={{ color: '#666', fontSize: '0.8rem' }}>
          {searchTerm ? 'No candidates found matching the search criteria': 'No candidates saved'}
          </Typography>
        </Box>
      ) : (
        <Box sx={{ 
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))',
          gap: 2.4,
          width: '100%'
        }}>
          {filteredCandidates.map((candidate) => (
            <CandidateCard 
              key={candidate._id} 
              candidate={candidate} 
              onDelete={handleDeleteCandidate}
            />
          ))}
        </Box>
      )}
    </Box>
  );
};

export default CandidatesPage; 