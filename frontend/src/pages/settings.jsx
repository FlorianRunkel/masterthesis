import React, { useState, useEffect } from 'react';
import { Box, Typography, TextField, Button, Alert, Card, List, ListItem, ListItemText, IconButton, Dialog, DialogTitle, DialogContent, DialogActions, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Paper, Switch, FormControlLabel } from '@mui/material';
import { useTheme } from '@mui/material/styles';
import { useMediaQuery } from '@mui/material';
import DeleteIcon from '@mui/icons-material/Delete';
import { API_BASE_URL } from '../api';

const SettingsPage = () => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const [form, setForm] = useState({ firstName: '', lastName: '', email: '', password: '', canViewExplanations: false });
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(null);
  const [error, setError] = useState(null);
  const [users, setUsers] = useState([]);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [userToDelete, setUserToDelete] = useState(null);
  const [deleteError, setDeleteError] = useState(null);
  const [deleteSuccess, setDeleteSuccess] = useState(null);
  const [editUsers, setEditUsers] = useState([]);
  const [updateSuccess, setUpdateSuccess] = useState(null);
  const [updateError, setUpdateError] = useState(null);

  useEffect(() => {
    const user = JSON.parse(localStorage.getItem('user'));
    if (user?.firstName !== 'admin') {
    }
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setSuccess(null);
    setError(null);
    setDeleteSuccess(null);
    try {
      const response = await fetch(`${API_BASE_URL}/api/create-user`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form)
      });
      const data = await response.json();
      if (response.ok) {
        setSuccess('User erfolgreich angelegt!');
        setForm({ firstName: '', lastName: '', email: '', password: '', canViewExplanations: false });
      } else {
        setError(data.error || 'Fehler beim Anlegen des Users.');
      }
    } catch (err) {
      setError('Serverfehler: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const fetchUsers = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/users`);
      const data = await res.json();
      console.log('User-API Antwort:', data);
      if (res.ok) {
        setUsers(Array.isArray(data) ? data : data.users || data.data || []);
      }
    } catch (e) {
      console.error(e);
    }
  };

  useEffect(() => {
    fetchUsers();
  }, []);

  useEffect(() => {
    if (success || deleteSuccess) {
      fetchUsers();
    }
  }, [success, deleteSuccess]);

  useEffect(() => {
    setEditUsers(users.map(u => ({ ...u, password: '' })));
  }, [users]);

  const handleDeleteClick = (user) => {
    setUserToDelete(user);
    setDeleteDialogOpen(true);
    setDeleteError(null);
    setDeleteSuccess(null);
  };

  const handleDeleteConfirm = async () => {
    if (!userToDelete) return;
    try {
      const res = await fetch(`${API_BASE_URL}/api/users/${userToDelete._id}`, { method: 'DELETE' });
      const data = await res.json();
      if (res.ok) {
        setDeleteSuccess('User erfolgreich gelöscht!');
        setDeleteDialogOpen(false);
        setUserToDelete(null);
        fetchUsers();
      } else {
        setDeleteError(data.error || 'Fehler beim Löschen.');
      }
    } catch (e) {
      setDeleteError('Serverfehler: ' + e.message);
    }
  };

  const handleDeleteCancel = () => {
    setDeleteDialogOpen(false);
    setUserToDelete(null);
    setDeleteError(null);
    setDeleteSuccess(null);
  };

  const handleEditChange = (idx, field, value) => {
    setEditUsers(prev => prev.map((u, i) => i === idx ? { ...u, [field]: value } : u));
  };

  const handleUpdateUser = async (user) => {
    setUpdateSuccess(null);
    setUpdateError(null);
    try {
      const res = await fetch(`${API_BASE_URL}/api/users/${user._id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          firstName: user.firstName,
          lastName: user.lastName,
          email: user.email,
          password: user.password || undefined,
          canViewExplanations: user.canViewExplanations || false
        })
      });
      const data = await res.json();
      if (res.ok) {
        fetchUsers();
        setUpdateSuccess('User erfolgreich aktualisiert!');
      } else {
        setUpdateError(data.error || 'Fehler beim Aktualisieren des Users.');
      }
    } catch (e) {
      setUpdateError('Serverfehler: ' + e.message);
    }
  };

  return (
    <Box sx={{ fontSize: '0.88rem', fontFamily: 'inherit', px: isMobile ? 1 : 0 }}>
      <Typography variant="h1" sx={{
        fontSize: isMobile ? '1.2rem' : '2rem',
        fontWeight: 700,
        color: '#001242',
        mb: isMobile ? 1.5 : 2
      }}>
        Admin Panel
      </Typography>
      <Typography sx={{
        color: '#666',
        mb: isMobile ? 2 : 4,
        fontSize: isMobile ? '0.88rem' : '0.88rem',
        maxWidth: isMobile ? '100%' : '800px'
      }}>
        Here you can create a new user account for the application.
      </Typography>
      <Box sx={{ bgcolor: '#fff', borderRadius: 2, boxShadow: '0 2px 8px rgba(0,0,0,0.05)', p: isMobile ? 2 : 4, maxWidth: isMobile ? '100%' : 1200, mb: isMobile ? 2 : 4, border: '1px solid #f0f0f0' }}>
        <Typography variant="h2" sx={{ fontSize: isMobile ? '1rem' : '1.15rem', fontWeight: 700, color: '#001242', mb: isMobile ? 2 : 3 }}>Create User</Typography>
        <form onSubmit={handleSubmit}>
          <Box sx={{ display: 'flex', gap: isMobile ? 1.2 : 2.4, flexDirection: isMobile ? 'column' : 'row', mb: 2 }}>
            <Box sx={{ flex: 1, minWidth: 0 }}>
              <Typography sx={{ fontWeight: 700, mb: 0.5, color: '#222', fontSize: '0.88rem' }}>First Name *</Typography>
              <TextField name="firstName" value={form.firstName} placeholder="First Name" required
                onChange={e => setForm(f => ({ ...f, firstName: e.target.value }))}
                sx={{
                  width: "100%",
                  '& .MuiOutlinedInput-root': { fontSize: '0.88rem', minHeight: isMobile ? '36px' : '38px', maxHeight: isMobile ? '36px' : '38px', bgcolor: '#fff', borderRadius: '8px', },
                  input: { fontSize: '0.88rem', height: isMobile ? '36px' : '38px', padding: '0 12px' },
                }}
              />
            </Box>
            <Box sx={{ flex: 1, minWidth: 0 }}>
              <Typography sx={{ fontWeight: 700, mb: 0.5, color: '#222', fontSize: '0.88rem' }}>Last Name *</Typography>
              <TextField name="lastName" value={form.lastName} placeholder="Last Name" required
                onChange={e => setForm(f => ({ ...f, lastName: e.target.value }))}
                sx={{
                  width: "100%",
                  '& .MuiOutlinedInput-root': { fontSize: '0.88rem', minHeight: isMobile ? '36px' : '38px', maxHeight: isMobile ? '36px' : '38px', bgcolor: '#fff', borderRadius: '8px', },
                  input: { fontSize: '0.88rem', height: isMobile ? '36px' : '38px', padding: '0 12px' },
                }}
              />
            </Box>
          </Box>
          <Box sx={{ mb: 2 }}>
            <Typography sx={{ fontWeight: 700, mb: 0.5, color: '#222', fontSize: '0.88rem' }}>E-Mail *</Typography>
            <TextField name="email" type="email" value={form.email} placeholder="E-Mail" required
              onChange={e => setForm(f => ({ ...f, email: e.target.value }))}
              sx={{
                width: "100%",
                '& .MuiOutlinedInput-root': { fontSize: '0.88rem', minHeight: isMobile ? '36px' : '38px', maxHeight: isMobile ? '36px' : '38px', bgcolor: '#fff', borderRadius: '8px', },
                input: { fontSize: '0.88rem', height: isMobile ? '36px' : '38px', padding: '0 12px' },
              }}
            />
          </Box>
          <Box sx={{ mb: 2 }}>
            <Typography sx={{ fontWeight: 700, mb: 0.5, color: '#222', fontSize: '0.88rem' }}>Password *</Typography>
            <TextField name="password" type="password" value={form.password} placeholder="Password" required
              onChange={e => setForm(f => ({ ...f, password: e.target.value }))}
              sx={{
                width: "100%",
                '& .MuiOutlinedInput-root': { fontSize: '0.88rem', minHeight: isMobile ? '36px' : '38px', maxHeight: isMobile ? '36px' : '38px', bgcolor: '#fff', borderRadius: '8px', },
                input: { fontSize: '0.88rem', height: isMobile ? '36px' : '38px', padding: '0 12px' },
              }}
            />
          </Box>
          <FormControlLabel
            control={
              <Switch
                checked={form.canViewExplanations}
                onChange={e => setForm(f => ({...f, canViewExplanations: e.target.checked}))}
              />
            }
            label="Allow user to view explanations"
            labelPlacement="end"
            sx={{ mb: 2, color: '#222', '& .MuiFormControlLabel-label': { fontSize: '0.88rem' } }}
          />
          <Box sx={{ display: 'flex', justifyContent: isMobile ? 'center' : 'left', mt: 2 }}>
            <Button type="submit" variant="contained" disabled={loading} sx={{ fontWeight: 800, py: 1, px: 4, borderRadius: '6px', fontSize: isMobile ? '0.95rem' : '1rem', letterSpacing: 0.5, background: '#EB7836', color: '#fff', boxShadow: 'none', textTransform: 'none', minWidth: 140, maxWidth: 260, height: isMobile ? '38px' : '42px', '&:hover': { background: '#EB7836', color: '#fff' } }}>CREATE USER</Button>
          </Box>
        </form>
        {success && <Alert severity="success" sx={{ mt: 2 }}>{success}</Alert>}
        {error && <Alert severity="error" sx={{ mt: 2 }}>{error}</Alert>}
      </Box>
      <Box sx={{ bgcolor: '#fff', borderRadius: 2, boxShadow: '0 2px 8px rgba(0,0,0,0.05)', p: isMobile ? 2 : 4, maxWidth: isMobile ? '100%' : 1200, mb: isMobile ? 2 : 4, border: '1px solid #f0f0f0' }}>
        <Typography variant="h2" sx={{ fontSize: isMobile ? '1rem' : '1.15rem', fontWeight: 700, color: '#001242', mb: isMobile ? 2 : 3 }}>
          Edit Users
        </Typography>
        {updateSuccess && <Alert severity="success" sx={{ mb: 2 }}>{updateSuccess}</Alert>}
        {updateError && <Alert severity="error" sx={{ mb: 2 }}>{updateError}</Alert>}
        {isMobile ? (
          <Box>
            {editUsers.map((user, idx) => (
              <Card key={user._id} sx={{ mb: 2.4, p: 2.4, borderRadius: 2, boxShadow: 1, border: '1px solid #fff', bgcolor: '#fff' }}>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                  <Box>
                    <Typography sx={{ fontWeight: 700, color: '#000000', fontSize: '0.88rem', mb: 0.5 }}>First Name</Typography>
                    <TextField
                      value={user.firstName}
                      onChange={e => handleEditChange(idx, 'firstName', e.target.value)}
                      size="small"
                      sx={{ fontSize: '0.88rem', '& .MuiInputBase-input': { fontSize: '0.88rem', py: 1.2 }, bgcolor: '#fff', borderRadius: '8px' }}
                      fullWidth
                    />
                  </Box>
                  <Box>
                    <Typography sx={{ fontWeight: 700, color: '#000000', fontSize: '0.88rem', mb: 0.5 }}>Last Name</Typography>
                    <TextField
                      value={user.lastName}
                      onChange={e => handleEditChange(idx, 'lastName', e.target.value)}
                      size="small"
                      sx={{ fontSize: '0.88rem', '& .MuiInputBase-input': { fontSize: '0.88rem', py: 1.2 }, bgcolor: '#fff', borderRadius: '8px' }}
                      fullWidth
                    />
                  </Box>
                  <Box>
                    <Typography sx={{ fontWeight: 700, color: '#000000', fontSize: '0.88rem', mb: 0.5 }}>E-Mail</Typography>
                    <TextField
                      value={user.email}
                      onChange={e => handleEditChange(idx, 'email', e.target.value)}
                      size="small"
                      sx={{ fontSize: '0.88rem', '& .MuiInputBase-input': { fontSize: '0.88rem', py: 1.2 }, bgcolor: '#fff', borderRadius: '8px' }}
                      fullWidth
                    />
                  </Box>
                  <Box>
                    <Typography sx={{ fontWeight: 700, color: '#000000', fontSize: '0.88rem', mb: 0.5 }}>Password</Typography>
                    <TextField
                      value={user.password}
                      onChange={e => handleEditChange(idx, 'password', e.target.value)}
                      size="small"
                      type="password"
                      placeholder="Neues Passwort"
                      sx={{ fontSize: '0.88rem', '& .MuiInputBase-input': { fontSize: '0.88rem', py: 1.2 }, bgcolor: '#fff', borderRadius: '8px' }}
                      fullWidth
                    />
                  </Box>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={user.canViewExplanations || false}
                        onChange={(e) => handleEditChange(idx, 'canViewExplanations', e.target.checked)}
                        disabled={user.uid === 'UID001'}
                      />
                    }
                    label="Show Explanations"
                    labelPlacement="start"
                    sx={{ mt: 1, justifyContent: 'space-between', ml: 0, '& .MuiFormControlLabel-label': { fontSize: '0.88rem' } }}
                  />
                  <Button
                    variant="contained"
                    color="primary"
                    onClick={() => handleUpdateUser(user)}
                    disabled={user.uid === 'UID001'}
                    sx={{ fontWeight: 700, fontSize: '1rem', borderRadius: '8px', px: 1.5, py: 1.2, minWidth: 60, height: '40px', textTransform: 'none', background: '#001242', '&:hover': { background: '#001242' }, mt: 2, boxShadow: '0 2px 8px rgba(33,118,174,0.08)' }}
                    fullWidth
                  >
                    SAVE
                  </Button>
                </Box>
              </Card>
            ))}
          </Box>
        ) : (
          <TableContainer component={Paper} sx={{maxHeight: '450px', boxShadow: 'none', borderRadius: 2, border: '1px #e0e0e0', width: '100%'}}>
            <Table stickyHeader size="medium">
              <TableHead>
                <TableRow>
                  <TableCell sx={{ fontWeight: 700, fontSize: '0.88rem', color: '#222', background: '#fafbfc', }}>First Name</TableCell>
                  <TableCell sx={{ fontWeight: 700, fontSize: '0.88rem', color: '#222', background: '#fafbfc' }}>Last Name</TableCell>
                  <TableCell sx={{ fontWeight: 700, fontSize: '0.88rem', color: '#222', background: '#fafbfc' }}>E-Mail</TableCell>
                  <TableCell sx={{ fontWeight: 700, fontSize: '0.88rem', color: '#222', background: '#fafbfc' }}>Password</TableCell>
                  <TableCell sx={{ fontWeight: 700, fontSize: '0.88rem', color: '#222', background: '#fafbfc'}}>Explanations</TableCell>
                  <TableCell sx={{ fontWeight: 700, fontSize: '0.88rem', color: '#222', background: '#fafbfc'}}>Action</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {editUsers.filter(user => user && user._id).map((user, idx) => (
                  <TableRow key={user._id} sx={{ height: 64, verticalAlign: 'middle'}}>
                    <TableCell sx={{ verticalAlign: 'middle' }}>
                      <TextField
                        value={user.firstName}
                        onChange={e => handleEditChange(idx, 'firstName', e.target.value)}
                        size="small"
                        sx={{ fontSize: '0.88rem', '& .MuiInputBase-input': { fontSize: '0.88rem', py: 1.2, height: '36px', boxSizing: 'border-box', display: 'flex', alignItems: 'center' } }}
                        inputProps={{ style: { height: '36px', padding: '0 12px', display: 'flex', alignItems: 'center' } }}
                      />
                    </TableCell>
                    <TableCell sx={{ verticalAlign: 'middle' }}>
                      <TextField
                        value={user.lastName}
                        onChange={e => handleEditChange(idx, 'lastName', e.target.value)}
                        size="small"
                        sx={{ fontSize: '0.88rem', '& .MuiInputBase-input': { fontSize: '0.88rem', py: 1.2, height: '36px', boxSizing: 'border-box', display: 'flex', alignItems: 'center' } }}
                        inputProps={{ style: { height: '36px', padding: '0 12px', display: 'flex', alignItems: 'center' } }}
                      />
                    </TableCell>
                    <TableCell sx={{ verticalAlign: 'middle' }}>
                      <TextField
                        value={user.email}
                        onChange={e => handleEditChange(idx, 'email', e.target.value)}
                        size="small"
                        sx={{ fontSize: '0.88rem', '& .MuiInputBase-input': { fontSize: '0.88rem', py: 1.2, height: '36px', boxSizing: 'border-box', display: 'flex', alignItems: 'center' } }}
                        inputProps={{ style: { height: '36px', padding: '0 12px', display: 'flex', alignItems: 'center' } }}
                      />
                    </TableCell>
                    <TableCell sx={{ verticalAlign: 'middle' }}>
                      <TextField
                        value={user.password}
                        onChange={e => handleEditChange(idx, 'password', e.target.value)}
                        size="small"
                        type="password"
                        placeholder="Neues Passwort"
                        sx={{ fontSize: '0.88rem', '& .MuiInputBase-input': { fontSize: '0.88rem', py: 1.2, height: '36px', boxSizing: 'border-box', display: 'flex', alignItems: 'center' } }}
                        inputProps={{ style: { height: '36px', padding: '0 12px', display: 'flex', alignItems: 'center' } }}
                      />
                    </TableCell>
                    <TableCell align="center" sx={{ verticalAlign: 'middle' }}>
                      <Switch
                        checked={user.canViewExplanations || false}
                        onChange={(e) => handleEditChange(idx, 'canViewExplanations', e.target.checked)}
                        disabled={user.uid === 'UID001'}
                      />
                    </TableCell>
                    <TableCell sx={{ verticalAlign: 'middle' }}>
                      <Button
                        variant="contained"
                        color="primary"
                        onClick={() => handleUpdateUser(user)}
                        disabled={user.uid === 'UID001'}
                        sx={{ fontWeight: 700, fontSize: '0.88rem', borderRadius: '6px', px: 2.5, py: 1, minWidth: 70, height: '36px', textTransform: 'none', background: '#001242', '&:hover': { background: '#001242' } }}
                      >
                        SAVE
                      </Button>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        )}
      </Box>
      <Box sx={{ mb: isMobile ? 2 : 4 }}>
        <Card
          sx={{
            bgcolor: '#fff',
            borderRadius: 2,
            boxShadow: '0 2px 8px rgba(0,0,0,0.05)',
            p: isMobile ? 2 : 4,
            maxWidth: isMobile ? '100%' : 1200,
            maxHeight: 400,
            overflowY: 'auto',
            mb: 2,
            border: '1px solid #f0f0f0'
          }}
        >
          <Typography variant="h2" sx={{ fontSize: isMobile ? '1rem' : '1.15rem', fontWeight: 700, color: '#001242', mb: isMobile ? 2 : 3 }}>
            Delete User
          </Typography>
          <List>
            {users.length === 0 && (
              <ListItem>
                <ListItemText primary="No users found." />
              </ListItem>
            )}
            {users.map((user) => (
              <ListItem
                key={user._id}
                divider
                secondaryAction={
                  user.uid === 'UID001' ? (
                    <IconButton edge="end" disabled sx={{ color: '#bbb' }}>
                      <DeleteIcon />
                    </IconButton>
                  ) : (
                    <IconButton edge="end" color="error" onClick={() => handleDeleteClick(user)} sx={{ color: '#e74c3c', '&:hover': { color: '#c0392b' } }}>
                      <DeleteIcon />
                    </IconButton>
                  )
                }
              >
                <ListItemText
                  primary={
                    <Typography sx={{ fontWeight: 600, color: '#222', fontSize: '0.88rem' }}>
                      {user.firstName} {user.lastName}
                    </Typography>
                  }
                  secondary={
                    <Typography sx={{ color: '#666', fontSize: '0.88rem' }}>
                      {user.email}
                      {user.uid === 'UID001' ? ' (Admin, cannot delete)' : ''}
                    </Typography>
                  }
                />
              </ListItem>
            ))}
          </List>
        </Card>
        <Dialog open={deleteDialogOpen} onClose={handleDeleteCancel} maxWidth="xs" fullWidth>
          <DialogTitle>Delete User</DialogTitle>
          <DialogContent>
            <Typography>Are you sure you want to delete user <b>{userToDelete?.firstName} {userToDelete?.lastName}</b>?</Typography>
            {deleteError && <Alert severity="error" sx={{ mt: 2 }}>{deleteError}</Alert>}
          </DialogContent>
          <DialogActions>
            <Button onClick={handleDeleteCancel} color="inherit">Cancel</Button>
            <Button onClick={handleDeleteConfirm} color="error" variant="contained">Delete</Button>
          </DialogActions>
        </Dialog>
      </Box>
    </Box>
  );
};

export default SettingsPage; 