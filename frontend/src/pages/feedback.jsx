import React, { useState, useEffect } from 'react';
import { Box, Typography, TextField, Button, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Paper, Radio, Select, MenuItem, IconButton, useMediaQuery } from '@mui/material';
import { useTheme } from '@mui/material/styles';
import DeleteIcon from '@mui/icons-material/Delete';
import { API_BASE_URL } from '../api';
import axios from 'axios';

const prognoseHeaders = ['Model type', 'Model prediction', 'Your assessment', 'Comment'];
const modelOptions = ['GRU', 'XGBoost', 'TFT'];
const ratingCriteria = [
    'Relevance and realism of predictions',
    'Transparency of model decisions',
    'Usefulness for daily work in Active Sourcing',
    'Trustworthiness of the AI recommendations',
    'Likelihood of future usage in real scenarios',
    'Overall impression and satisfaction'
  ];

const FeedbackPage = () => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  // Initialisiere State mit localStorage-Werten oder Standardwerten
  const [freeText, setFreeText] = useState(() => {
    const saved = localStorage.getItem('feedback_freeText');
    return saved || '';
  });
  
  const [prognoseBewertung, setPrognoseBewertung] = useState(() => {
    const saved = localStorage.getItem('feedback_prognoseBewertung');
    return saved ? JSON.parse(saved) : [{ modell: '', prognose: '', echt: '', bemerkung: '' }];
  });
  
  const [bewertungsskala, setBewertungsskala] = useState(() => {
    const saved = localStorage.getItem('feedback_bewertungsskala');
    return saved ? JSON.parse(saved) : Array(ratingCriteria.length).fill(3);
  });
  
  const [explanationFeedback, setExplanationFeedback] = useState(() => {
    const saved = localStorage.getItem('feedback_explanationFeedback');
    return saved ? JSON.parse(saved) : {};
  });

  const [success, setSuccess] = useState(false);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const user = JSON.parse(localStorage.getItem('user'));
  const canViewExplanations = user?.canViewExplanations;
  const explanationQuestionsNo = [
    { key: 'wantFeatureImportance', label: "Would it make the result easier to understand if the system showed you what mattered most in making the decision?" },
    { key: 'wantMoreExplainability', label: "Would you trust the result more if the system explained how it came to this decision?" }
  ];
  const explanationQuestionsYes = [
    { key: 'explanationHelpful', label: "Did the explanation make it clearer how the system came to this result?" },
    { key: 'featureImportanceUseful', label: "Was it helpful to see what mattered most in the system’s decision?" },
    { key: 'lessTrustWithoutExplanation', label: "If there had been no explanation at all, do you think you would have trusted the result in the same way?" }
  ];

  const handlePrognoseChange = (idx, field, value) => {
    const updated = prognoseBewertung.map((row, i) => i === idx ? { ...row, [field]: value } : row);
    setPrognoseBewertung(updated);
  };

  const addPrognoseRow = () => {
    setPrognoseBewertung([...prognoseBewertung, { modell: '', prognose: '', echt: '', bemerkung: '' }]);
  };

  const handleBewertungChange = (idx, value) => {
    const updated = [...bewertungsskala];
    updated[idx] = value;
    setBewertungsskala(updated);
  };

  const handleExplanationFeedback = (key, value) => {
    setExplanationFeedback(prev => ({ ...prev, [key]: value }));
  };

  // Speichere alle Änderungen automatisch im localStorage
  useEffect(() => {
    localStorage.setItem('feedback_freeText', freeText);
  }, [freeText]);

  useEffect(() => {
    localStorage.setItem('feedback_prognoseBewertung', JSON.stringify(prognoseBewertung));
  }, [prognoseBewertung]);

  useEffect(() => {
    localStorage.setItem('feedback_bewertungsskala', JSON.stringify(bewertungsskala));
  }, [bewertungsskala]);

  useEffect(() => {
    localStorage.setItem('feedback_explanationFeedback', JSON.stringify(explanationFeedback));
  }, [explanationFeedback]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setSuccess(false);
    setError('');
    try {
      const uid = user?.uid;
      const res = await axios.post(`${API_BASE_URL}/api/feedback`, { freeText, prognoseBewertung, bewertungsskala, explanationFeedback }, {
        headers: { 'X-User-Uid': uid }
      });
      setSuccess(true);
      
      // Nach erfolgreichem Submit alle localStorage-Daten löschen
      localStorage.removeItem('feedback_freeText');
      localStorage.removeItem('feedback_prognoseBewertung');
      localStorage.removeItem('feedback_bewertungsskala');
      localStorage.removeItem('feedback_explanationFeedback');
      
      // Formular zurücksetzen
      setFreeText('');
      setPrognoseBewertung([{ modell: '', prognose: '', echt: '', bemerkung: '' }]);
      setBewertungsskala(Array(ratingCriteria.length).fill(3));
      setExplanationFeedback({});
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
          <Typography variant="h2" sx={{ fontSize: '1.15rem', fontWeight: 700, mb: 1, color: '#001242' }}>Evaluation Criteria</Typography>
          <Typography sx={{ fontSize: '0.88rem', color: '#666', mb: 1.5 }}>
            Please evaluate each criterion on a scale from 1 to 5, where 1 means very poor or not helpful, and 5 means excellent and extremely useful.
          </Typography>
          <TableContainer component={Paper} sx={{ mb: 1, boxShadow: 0 }}>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell sx={{ fontWeight: 700, color: '#001242', fontSize: '0.88rem', bgcolor: '#fff' }}>Criterion</TableCell>
                  {[1,2,3,4,5].map(val => <TableCell key={val} align="center" sx={{ fontWeight: 700, color: '#001242', fontSize: '1rem', bgcolor: '#fff' }}>{val}</TableCell>)}
                </TableRow>
              </TableHead>
              <TableBody>
                {ratingCriteria.map((criterion, idx) => (
                  <TableRow key={criterion} sx={{ fontSize: '0.88rem', mb: 1, color: '#111'}}>
                    <TableCell sx={{ fontSize: '0.88rem', mb: 1, color: '#111'}}>{criterion}</TableCell>
                    {[1,2,3,4,5].map(val => (
                      <TableCell key={val} align="center">
                        <Radio
                          checked={bewertungsskala[idx] === val}
                          onChange={() => handleBewertungChange(idx, val)}
                          value={val}
                          name={`rating-${idx}`}
                          color="primary"
                          sx={{
                            '&.Mui-checked': {
                              color: '#001242',
                            },
                          }}
                          size="small"
                        />
                      </TableCell>
                    ))}
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </Box>
        <Box sx={{ bgcolor: '#fff', borderRadius: 3, p: { xs: 2, sm: 3 }, mb: 4, boxShadow: '0 2px 8px rgba(0,0,0,0.05)'   }}>
          <Typography variant="h2" sx={{ fontSize: '1.15rem', fontWeight: 700, mb: 2, color: '#001242' }}>
            Explainability Feedback
          </Typography>
          {(canViewExplanations ? explanationQuestionsYes : explanationQuestionsNo).map((q, idx, arr) => (
            <Box key={q.key} sx={{ mb: idx < arr.length - 1 ? 2 : 0 }}>
              <Typography sx={{fontSize: '0.88rem', mb: 1, color: '#111',}}>{q.label}</Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 3, mb: 1 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <input
                    type="radio"
                    id={`${q.key}-yes`}
                    name={q.key}
                    checked={explanationFeedback[q.key] === 'yes'}
                    onChange={() => handleExplanationFeedback(q.key, 'yes')}
                    style={{ accentColor: '#001242', width: 14, height: 14 }}
                  />
                  <label htmlFor={`${q.key}-yes`} style={{ marginRight: 16, fontWeight: 500, fontSize: '0.8rem', color: '#111', }}>YES</label>
                </Box>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <input
                    type="radio"
                    id={`${q.key}-no`}
                    name={q.key}
                    checked={explanationFeedback[q.key] === 'no'}
                    onChange={() => handleExplanationFeedback(q.key, 'no')}
                    style={{ accentColor: '#001242', width: 14, height: 14 }}
                  />
                  <label htmlFor={`${q.key}-no`} style={{ fontWeight: 500, fontSize: '0.88rem', color: '#111', }}>NO</label>
                </Box>
              </Box>
              {idx < arr.length - 1 && <Box sx={{ borderBottom: '1px solid #e0e0e0', my: 2 }} />}
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