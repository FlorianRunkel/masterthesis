import React, { useState, useEffect } from 'react';
import { Box, Typography, TextField, Button, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Paper, Radio, Select, MenuItem, IconButton, useMediaQuery } from '@mui/material';
import { useTheme } from '@mui/material/styles';
import DeleteIcon from '@mui/icons-material/Delete';
import { API_BASE_URL } from '../api';
import axios from 'axios';

const prognoseHeaders = ['Model type', 'Model prediction', 'Your assessment', 'Comment'];
const modelOptions = ['GRU', 'XGBoost', 'TFT'];

/*
group A Questions (Predictions only)
*/
const controlGroupQuestions = [
  { category: 'Comprehensibility & Interpretability', questions: [
    'The system’s predictions about candidate job-switching readiness seemed realistic.',
    'The predictions were relevant for prioritizing candidates in Active Sourcing.',
  ]},
  { category: 'Confidence in Predictions', questions: [
    'I trusted the system’s predictions when deciding which candidates to approach.',
    'The recommendations gave me enough confidence to base sourcing decisions on them.'
  ]},
  { category: 'Usability for Recruiting', questions: [
    'The system’s predictions were easy to interpret without further explanation.',
    'The predictions helped me to structure the candidate selection process more efficiently.'
  ]},  
  { category: 'Integration of Human Expertise and AI Support', questions: [
    'The system’s predictions supported me in combining them with my own recruiting expertise.',
    'The system turned out to be a valuable complement to my own judgment.'
  ]},
  { category: 'Perceived Value & Intention to Use', questions: [
    'I can imagine using such a prediction system in my daily recruiting activities.',
    'The system would help me to improve the effectiveness of my sourcing decisions.',
  ]}
];

/*
group B Questions (Predictions + Explanations)
*/
const experimentalGroupQuestions = [
  { category: 'Comprehensibility & Interpretability', questions: [
    'The explanations made it clear why a candidate was predicted as more or less likely to switch jobs.',
    'The explanations increased my understanding of how the system generated its predictions.',
  ]},
  { category: 'Confidence in Predictions', questions: [
    'The explanations strengthened my confidence in the reliability of the predictions.',
    'The presence of explanations made me more willing to act on the system\'s recommendations.'
  ]},
  { category: 'Usability for Recruiting', questions: [
    'The combination of predictions and explanations was straightforward and clear to understand.',
    'The explanations improved my ability to identify which candidates should be prioritized in Active Sourcing.'
  ]},
  { category: 'Integration of Human Expertise and AI Support', questions: [
    'The explanations supported me in combining the system\'s predictions with my own recruiting expertise.',
    'The system turned out to be a valuable complement to my own judgment.'
  ]},
  { category: 'Perceived Value & Intention to Use', questions: [
    'I could imagine integrating such a system with explanations into my daily recruiting workflow.',
    'The explanations provided added value compared to predictions alone.'
  ]}
];

const FeedbackPage = () => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const [freeText, setFreeText] = useState(() => {
    const saved = localStorage.getItem('feedback_freeText');
    return saved || '';
  });

  const [prognoseBewertung, setPrognoseBewertung] = useState(() => {
    const saved = localStorage.getItem('feedback_prognoseBewertung');
    return saved ? JSON.parse(saved) : [{ modell: '', prognose: '', echt: '', bemerkung: '' }];
  });

  const [success, setSuccess] = useState(false);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const user = JSON.parse(localStorage.getItem('user'));
  const canViewExplanations = user?.canViewExplanations;

  const currentQuestions = canViewExplanations ? experimentalGroupQuestions : controlGroupQuestions;
  const totalQuestions = currentQuestions.reduce((sum, category) => sum + category.questions.length, 0);

  const [bewertungsskala, setBewertungsskala] = useState(() => {
    const saved = localStorage.getItem('feedback_bewertungsskala');
    return saved ? JSON.parse(saved) : Array(totalQuestions).fill(3);
  });

  const handlePrognoseChange = (idx, field, value) => {
    const updated = prognoseBewertung.map((row, i) => i === idx ? { ...row, [field]: value } : row);
    setPrognoseBewertung(updated);
  };

  const addPrognoseRow = () => {
    setPrognoseBewertung([...prognoseBewertung, { modell: '', prognose: '', echt: '', bemerkung: '' }]);
  };

  const handleBewertungChange = (questionIdx, value) => {
    const updated = [...bewertungsskala];
    updated[questionIdx] = value;
    setBewertungsskala(updated);
  };

  /*
  Save feedback.
  */
  useEffect(() => {
    localStorage.setItem('feedback_freeText', freeText);
  }, [freeText]);

  useEffect(() => {
    localStorage.setItem('feedback_prognoseBewertung', JSON.stringify(prognoseBewertung));
  }, [prognoseBewertung]);

  useEffect(() => {
    localStorage.setItem('feedback_bewertungsskala', JSON.stringify(bewertungsskala));
  }, [bewertungsskala]);

  /*
  Handle submit.
  */
  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setSuccess(false);
    setError('');
    try {
      const uid = user?.uid;
      await axios.post(`${API_BASE_URL}/api/feedback`, { freeText, prognoseBewertung, bewertungsskala }, {
        headers: { 'X-User-Uid': uid }
      });
      setSuccess(true);

      localStorage.removeItem('feedback_freeText');
      localStorage.removeItem('feedback_prognoseBewertung');
      localStorage.removeItem('feedback_bewertungsskala');

      setFreeText('');
      setPrognoseBewertung([{ modell: '', prognose: '', echt: '', bemerkung: '' }]);
      setBewertungsskala(Array(totalQuestions).fill(3));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box sx={{ p: 0, m: 0 }}>
        <Typography variant="h1" sx={{ 
        fontSize: isMobile ? '1.8rem' : '2.5rem', 
        fontWeight: 700, 
        color: '#001242', 
        mb: 2 
      }}>
        Feedback & Evaluation
      </Typography>
      <Typography sx={{ 
        color: '#666', 
        mb: 4, 
        fontSize: isMobile ? '0.9rem' : '1rem', 
      }}>
        Please share your experience with the dashboard and rate the predictions and user experience. Your feedback helps me to improve the system!
      </Typography>
      <form onSubmit={handleSubmit}>
        <Box sx={{ bgcolor: '#fff', borderRadius: 3, p: { xs: 2, sm: 3 }, mb: 4, boxShadow: '0 2px 8px rgba(0,0,0,0.05)' }}>
          <Typography variant="h2" sx={{ fontSize: '1.15rem', fontWeight: 700, mb: 1.5, color: '#001242' }}>General feedback</Typography>
          <TextField
            placeholder="Your feedback, suggestions, experience..."
            multiline
            minRows={4}
            fullWidth
            value={freeText}
            onChange={e => setFreeText(e.target.value)}
            sx={{ mb: 1.5, bgcolor: '#fff', borderRadius: 2, fontSize: '0.8rem' }}
          />
        </Box>
        <Box sx={{ bgcolor: '#fff', borderRadius: 3, p: { xs: 2, sm: 3 }, mb: 4, boxShadow: '0 2px 8px rgba(0,0,0,0.05)' }}>
          <Typography variant="h2" sx={{ fontSize: '1.15rem', fontWeight: 700, mb: 1, color: '#001242' }}>Candidate input & prediction evaluation</Typography>
          <Typography sx={{ fontSize: '0.88rem', color: '#666', mb: 1.5 }}>
            Please use this section to document individual prediction cases. Select the model you used, enter the prediction it produced, and compare it with your own assessment or ground truth. Feel free to add any comments regarding the accuracy, surprising results, or potential reasons for discrepancies.
          </Typography>
          {isMobile ? (
            <Box>
              {prognoseBewertung.map((row, idx) => (
                <Box key={idx} sx={{ mb: 3, p: 2, border: '1px solid #fff', borderRadius: 2, position: 'relative', bgcolor: '#fff' }}>
                  <Box sx={{ mb: 1 }}>
                    <Typography sx={{ fontWeight: 600, fontSize: '0.98rem', mb: 0.5 }}>Model type</Typography>
                    <Select
                      value={row.modell}
                      onChange={e => handlePrognoseChange(idx, 'modell', e.target.value)}
                      displayEmpty
                      fullWidth
                      size="small"
                      sx={{ bgcolor: '#fff', borderRadius: 1 }}
                      renderValue={selected => selected || 'Select used model'}
                    >
                      <MenuItem value=""><em>Select model</em></MenuItem>
                      {modelOptions.map(opt => (
                        <MenuItem key={opt} value={opt}>{opt}</MenuItem>
                      ))}
                    </Select>
                  </Box>
                  <Box sx={{ mb: 1 }}>
                    <Typography sx={{ fontWeight: 600, fontSize: '0.98rem', mb: 0.5 }}>Model prediction</Typography>
                    <TextField value={row.prognose} onChange={e => handlePrognoseChange(idx, 'prognose', e.target.value)} size="small" placeholder="Prediction" fullWidth />
                  </Box>
                  <Box sx={{ mb: 1 }}>
                    <Typography sx={{ fontWeight: 600, fontSize: '0.98rem', mb: 0.5 }}>Your assessment</Typography>
                    <TextField value={row.echt} onChange={e => handlePrognoseChange(idx, 'echt', e.target.value)} size="small" placeholder="Your assessment" fullWidth />
                  </Box>
                  <Box>
                    <Typography sx={{ fontWeight: 600, fontSize: '0.98rem', mb: 0.5 }}>Comment / Feedback</Typography>
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      <TextField value={row.bemerkung} onChange={e => handlePrognoseChange(idx, 'bemerkung', e.target.value)} size="small" placeholder="Comment" fullWidth multiline minRows={1} maxRows={3} />
                      {prognoseBewertung.length > 1 && (
                        <IconButton aria-label="delete row" onClick={() => {
                          setPrognoseBewertung(prognoseBewertung.filter((_, i) => i !== idx));
                        }} sx={{ ml: 1, color: 'error.main' }}>
                          <DeleteIcon />
                        </IconButton>
                      )}
                    </Box>
                  </Box>
                </Box>
              ))}
            </Box>
          ) : (
            <TableContainer component={Paper} sx={{
              mb: 2,
              bgcolor: '#fff',
              borderRadius: 1,
              boxShadow: 'none',
            }}>
              <Table size="small" sx={{
                borderCollapse: 'collapse',
                borderSpacing: 0,
                border: '0.5px solid #e0e0e0',
                boxShadow: 'none',
              }}>
                <TableHead>
                  <TableRow sx={{ bgcolor: '#fafbfc' }}>
                    {prognoseHeaders.map(h => (
                      <TableCell key={h} sx={{ fontWeight: 700, color: '#111111', fontSize: '1rem', borderTop: 'none', borderLeft: 'none', bgcolor: '#fafbfc' }}>
                        {h}
                      </TableCell>
                    ))}
                  </TableRow>
                </TableHead>
                <TableBody>
                  {prognoseBewertung.map((row, idx) => (
                    <TableRow key={idx}>
                      <TableCell sx={{ minWidth: 50, maxWidth: 70, borderLeft: 'none' }}>
                        <Select
                          value={row.modell}
                          onChange={e => handlePrognoseChange(idx, 'modell', e.target.value)}
                          displayEmpty
                          fullWidth
                          size="small"
                          sx={{ bgcolor: '#fff', borderRadius: 1 }}
                          renderValue={selected => selected || 'Select used model'}
                        >
                          <MenuItem value=""><em>Select model</em></MenuItem>
                          {modelOptions.map(opt => (
                            <MenuItem key={opt} value={opt}>{opt}</MenuItem>
                          ))}
                        </Select>
                      </TableCell>
                      <TableCell sx={{ minWidth: 120, maxWidth: 180 }}>
                        <TextField value={row.prognose} onChange={e => handlePrognoseChange(idx, 'prognose', e.target.value)} size="small" placeholder="Prediction" fullWidth />
                      </TableCell>
                      <TableCell sx={{ minWidth: 120, maxWidth: 180 }}>
                        <TextField value={row.echt} onChange={e => handlePrognoseChange(idx, 'echt', e.target.value)} size="small" placeholder="Your assessment" fullWidth />
                      </TableCell>
                      <TableCell sx={{ minWidth: 320 }}>
                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                          <TextField value={row.bemerkung} onChange={e => handlePrognoseChange(idx, 'bemerkung', e.target.value)} size="small" placeholder="Comment" fullWidth multiline minRows={1} maxRows={3} />
                          {prognoseBewertung.length > 1 && (
                            <IconButton aria-label="delete row" onClick={() => {
                              setPrognoseBewertung(prognoseBewertung.filter((_, i) => i !== idx));
                            }} sx={{ ml: 1, color: 'error.main' }}>
                              <DeleteIcon />
                            </IconButton>
                          )}
                        </Box>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          )}
          <Button
            onClick={addPrognoseRow}
            variant="outlined"
            sx={{
              mb: 1,
              color: '#666',
              borderColor: '#666',
              borderRadius: 3,
              fontWeight: 700,
              fontSize: '0.9rem',
              textTransform: 'none',
              px: 1.5,
              py: 0.5,
              background: 'transparent',
              '&:hover': {
                background: '#FFF3E6',
                borderColor: '#666',
              },
              minWidth: '100px'
            }}
          >
            + Add row
          </Button>
        </Box>
        <Box sx={{ bgcolor: '#fff', borderRadius: 3, p: { xs: 2, sm: 3 }, mb: 4, boxShadow: '0 2px 8px rgba(0,0,0,0.05)' }}>
          <Typography variant="h2" sx={{ fontSize: '1.15rem', fontWeight: 700, mb: 1, color: '#001242' }}>
            Evaluation Questionnaire
          </Typography>
          <Typography sx={{ fontSize: '0.88rem', color: '#666', mb: 2 }}>
            Please evaluate each statement on a scale from 1 to 5, where 1 means "Strongly Disagree" and 5 means "Strongly Agree".
          </Typography>
          
          {currentQuestions.map((category, categoryIdx) => (
            <Box key={category.category} sx={{ mb: categoryIdx < currentQuestions.length - 1 ? 3 : 0 }}>
              <Typography variant="h3" sx={{ fontSize: '1rem', fontWeight: 600, mb: 1.5, color: '#111' }}>
                {categoryIdx + 1}. {category.category}
              </Typography>
              
              {category.questions.map((question, questionIdx) => {
                const globalQuestionIdx = currentQuestions.slice(0, categoryIdx).reduce((sum, cat) => sum + cat.questions.length, 0) + questionIdx;
                return (
                  <Box key={globalQuestionIdx} sx={{ mb: 2, p: 2, bgcolor: '#fafbfc', borderRadius: 2, border: '1px solid #fafbfc' }}>
                    <Typography sx={{ fontSize: '0.9rem', mb: 1.5, color: '#111', fontWeight: 500 }}>
                      {question}
                    </Typography>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, flexWrap: 'wrap' }}>
                      <Typography sx={{ fontSize: '0.8rem', color: '#666', mr: 1 }}>Strongly Disagree</Typography>
                      {[1,2,3,4,5].map(val => (
                        <Box key={val} sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                          <Radio
                            checked={bewertungsskala[globalQuestionIdx] === val}
                            onChange={() => handleBewertungChange(globalQuestionIdx, val)}
                            value={val}
                            name={`rating-${globalQuestionIdx}`}
                            color="primary"
                            sx={{
                              '&.Mui-checked': {
                                color: '#001242',
                              },
                            }}
                            size="small"
                          />
                          <Typography sx={{ fontSize: '0.8rem', color: '#666', minWidth: '20px', textAlign: 'center' }}>
                            {val}
                          </Typography>
                        </Box>
                      ))}
                      <Typography sx={{ fontSize: '0.8rem', color: '#666', ml: 1 }}>Strongly Agree</Typography>
                    </Box>
                  </Box>
                );
              })}
            </Box>
          ))}
        </Box>
        <Button type="submit" disabled={loading} variant="contained" sx={{ bgcolor: '#EB7836', color: '#fff', fontWeight: 700, px: 4, py: 1.2, borderRadius: 2, fontSize: '1.08rem', mb: 2, boxShadow: '0 2px 8px rgba(0,0,0,0.05)', textTransform: 'none', letterSpacing: 0.2 }}>Submit feedback</Button>
        {success && <Typography sx={{ color: 'green', mt: 2, fontWeight: 600 }}>Feedback submitted successfully!</Typography>}
        {error && <Typography sx={{ color: 'red', mt: 2 }}>{error}</Typography>}
      </form>
    </Box>
  );
};

export default FeedbackPage; 