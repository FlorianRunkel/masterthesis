import React, { useState, useEffect } from 'react';
import { Box, Typography, TextField, InputAdornment, IconButton } from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import CandidateCard from '../components/candidates/CandidateCard';
import LoadingSpinner from '../components/LoadingSpinner';

const CandidatesPage = () => {
  const [candidates, setCandidates] = useState([]);
  const [filteredCandidates, setFilteredCandidates] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');

  useEffect(() => {
    const fetchCandidates = async () => {
      try {
        setLoading(true);
        const response = await fetch('http://localhost:5100/candidates');
        if (!response.ok) {
          throw new Error('Fehler beim Laden der Kandidaten');
        }
        const data = await response.json();
        setCandidates(data);
        setFilteredCandidates(data);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchCandidates();
  }, []);

  useEffect(() => {
    const filtered = candidates.filter(candidate => {
      const name = `${candidate.firstName || ''} ${candidate.lastName || ''}`.toLowerCase();
      return name.includes(searchTerm.toLowerCase()) || 
             candidate.linkedinProfile?.toLowerCase().includes(searchTerm.toLowerCase());
    });
    setFilteredCandidates(filtered);
  }, [searchTerm, candidates]);

  if (loading) {
    return <LoadingSpinner />;
  }

  if (error) {
    return (
      <Box sx={{ maxWidth: '1200px', margin: '0 auto', p: 4 }}>
        <Typography variant="h2" sx={{ mb: 3, color: '#1a1a1a', fontSize: '1.5rem', fontWeight: 600 }}>
          Fehler
        </Typography>
        <Typography sx={{ color: '#dc3545' }}>{error}</Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ maxWidth: '1200px', margin: '0 auto'}}>
      <Typography variant="h1" sx={{
        fontSize: '2.5rem',
        fontWeight: 700,
        color: '#1a1a1a',
        mb: 2
      }}>
        Kandidaten
      </Typography>

      <Typography sx={{
        color: '#666',
        mb: 4,
        fontSize: '1rem',
        maxWidth: '800px'
      }}>
        Hier können Sie alle gespeicherten Kandidaten einsehen.
        </Typography>

      <Box sx={{
        bgcolor: '#fff',
        borderRadius: '16px',
        p: '30px',
        boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
        mb: 4,
        width: '100%'
      }}>
        <Typography variant="h2" sx={{
          fontSize: '1.5rem',
          fontWeight: 600,
          color: '#1a1a1a',
          mb: 3
        }}>
          Kandidaten suchen
        </Typography>

        <TextField 
          placeholder="Nach Name oder LinkedIn-Profil suchen..."
          variant="outlined"
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          sx={{ 
            width: '100%',
            '& .MuiOutlinedInput-root': {
              bgcolor: '#fff',
              '& fieldset': {
                borderColor: '#e0e0e0',
                borderWidth: 1
              },
              '&:hover fieldset': {
                borderColor: '#1a1a1a'
              },
              '&.Mui-focused fieldset': {
                borderColor: '#1a1a1a'
              }
            }
          }}
          InputProps={{
            endAdornment: (
              <InputAdornment position="end">
                <IconButton>
                  <SearchIcon />
                </IconButton>
              </InputAdornment>
            )
          }}
        />
      </Box>

      {filteredCandidates.length === 0 ? (
        <Box sx={{
          bgcolor: '#fff',
          borderRadius: '16px',
          p: '30px',
          boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
          textAlign: 'center',
          width: '100%'
        }}>
          <Typography sx={{ color: '#666' }}>
            {searchTerm ? 'Keine Kandidaten gefunden, die den Suchkriterien entsprechen.' : 'Keine Kandidaten gespeichert.'}
          </Typography>
        </Box>
      ) : (
        <Box sx={{ 
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))',
          gap: 3,
          width: '100%'
        }}>
          {filteredCandidates.map((candidate) => (
            <CandidateCard 
              key={candidate._id} 
              candidate={candidate} 
            />
          ))}
        </Box>
      )}
    </Box>
  );
};

export default CandidatesPage; 